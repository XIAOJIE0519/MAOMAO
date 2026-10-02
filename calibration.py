"""Restore V5 event calibration on all rows with patient-disjoint selection.

Only the external calibration arrays are accepted. The prediction is
softmax((logits + class_bias) / temperature). The V5 positive-set likelihood
and 0.002 mean-square bias penalty are retained; bounded L-BFGS replaces Adam
to avoid hundreds of unnecessary full-cohort passes. No test scores select a
family, parameter, stopping rule, or threshold.
"""
import hashlib
import math
import time
import numpy as np
import torch
from scipy.optimize import minimize

PROTOCOL = "v5_positive_set_temperature_and_bias_full90_patient_selection_v2"
FAMILIES = ("raw", "temperature_only", "temperature_and_bias")
PENALTY = 0.002


def apply_calibration(logits, fit):
    bias = torch.as_tensor(fit.get("bias", [0.] * logits.shape[-1]),
                           dtype=logits.dtype, device=logits.device)
    return (logits + bias) / float(fit["temperature"])


def validate_fit(fit, rows, classes):
    """Check the saved calibration-only selection, refit and parameter contract."""
    assert fit['protocol']==PROTOCOL and fit['calibration_rows']==rows
    assert fit['test_labels_used_for_fit_or_selection'] is False
    assert fit['all_calibration_rows_used'] and not fit['model_weights_updated']
    assert fit['row_logit_shift_invariant'] and .15<=fit['temperature']<=6.
    assert len(fit['bias'])==classes and np.isfinite(fit['bias']).all()
    assert np.max(np.abs(fit['bias']))<=4.+1e-8
    split=fit['selection_split']
    assert split['patient_or_record_overlap']==0 and split['fit_groups']>0 and split['selection_groups']>0
    assert split['fit_rows']+split['selection_rows']==rows and min(split['fit_rows'],split['selection_rows'])>0
    trials=fit['selection_trials'];assert [x['family'] for x in trials]==list(FAMILIES)
    chosen=min(trials,key=lambda x:x['selection_nll'])
    if chosen['selection_nll']>=trials[0]['selection_nll']-1e-6:chosen=trials[0]
    assert fit['selected_family_before_full_refit_guard']==chosen['family']
    assert fit['raw_calibration']['rows']==fit['fitted_calibration']['rows']==rows
    guard=fit['fitted_calibration']['positive_set_nll']<=fit['raw_calibration']['positive_set_nll']+1e-7
    assert fit['full_refit_nll_guard_passed']==guard
    assert fit['family']==(chosen['family'] if guard else 'raw')
    assert fit['fitted_temperature_accepted']==(fit['family']!='raw')
    if fit['family']=='raw':assert fit['temperature']==1. and not np.any(fit['bias'])
    else:assert any(x['scope']=='all' and x['family']==fit['family'] and x['full_row_passes']>0 for x in fit['optimization_history'])
    return True


def selection_rows(groups, seed=42):
    groups = np.asarray(groups, dtype=np.int64)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("At least two calibration patient/record groups required")
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique)
    nselect = min(len(unique)-1, max(1, round(.2*len(unique))))
    selected = np.sort(order[:nselect])
    mask = np.isin(groups, selected)
    if not mask.any() or mask.all():
        raise ValueError("Empty patient-disjoint calibration fit/selection split")
    return mask, dict(seed=seed, fit_groups=len(unique)-nselect,
                      selection_groups=nselect, patient_or_record_overlap=0,
                      fit_rows=int((~mask).sum()), selection_rows=int(mask.sum()),
                      row_group_sha256=hashlib.sha256(groups.tobytes()).hexdigest(),
                      selection_group_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
                      definition="80:20 patient/record-proxy partition inside external 90%; final refit uses all external 90% rows")


class FullRows:
    def __init__(self, logits_path, labels_path, groups, device):
        self.z = np.load(logits_path, mmap_mode="r")
        self.y = np.load(labels_path, mmap_mode="r")
        if self.z.shape != self.y.shape or not len(self.y):
            raise ValueError("Calibration logits and labels differ")
        self.n, self.c = self.y.shape
        if len(groups) != self.n:
            raise ValueError("Calibration group alignment differs")
        self.select, self.split = selection_rows(groups)
        self.device = torch.device(device)
        self.zgpu = self.ygpu = None
        required = self.n*self.c*5
        if self.device.type == "cuda" and required < .70*torch.cuda.mem_get_info(self.device)[0]:
            self.zgpu = torch.empty(self.z.shape, dtype=torch.float32, device=self.device)
            self.ygpu = torch.empty(self.y.shape, dtype=torch.bool, device=self.device)
            for lo in range(0,self.n,65536):
                self.zgpu[lo:lo+65536] = torch.as_tensor(np.array(self.z[lo:lo+65536]), device=self.device)
                self.ygpu[lo:lo+65536] = torch.as_tensor(np.array(self.y[lo:lo+65536], dtype=bool), device=self.device)
        self.storage = "full FP32 logits and bool targets resident on GPU" if self.zgpu is not None else "complete memory-mapped rows streamed in chunks"
        print(f"Calibration: {self.n:,} rows, {self.c} classes; {self.storage}",flush=True)

    def chunks(self, scope):
        for lo in range(0,self.n,65536):
            hi=min(self.n,lo+65536)
            x = self.zgpu[lo:hi] if self.zgpu is not None else torch.as_tensor(np.array(self.z[lo:hi],dtype=np.float32),device=self.device)
            y = self.ygpu[lo:hi] if self.ygpu is not None else torch.as_tensor(np.array(self.y[lo:hi],dtype=bool),device=self.device)
            if scope != "all":
                m=self.select[lo:hi] if scope=="selection" else ~self.select[lo:hi]
                keep=torch.as_tensor(m,device=self.device)
                x,y=x[keep],y[keep]
            if len(x):
                if not bool(torch.isfinite(x).all()) or not bool(y.any(1).all()):
                    raise ValueError("Nonfinite calibration logits or empty target")
                # Exact row-shift invariance before temperature fitting.
                yield x-x.max(1,keepdim=True).values,y

    def active(self, scope):
        active=torch.zeros(self.c,dtype=torch.bool,device=self.device)
        for _,y in self.chunks(scope):active |= y.any(0)
        return active.cpu().numpy()

    @torch.inference_mode()
    def statistics(self, logt, bias, scope, gradient=False):
        t=math.exp(float(logt));b=torch.as_tensor(bias,dtype=torch.float32,device=self.device)
        n=0;total=torch.zeros((),dtype=torch.float64,device=self.device)
        gb=torch.zeros(self.c,dtype=torch.float64,device=self.device)
        gt=torch.zeros((),dtype=torch.float64,device=self.device)
        br=torch.zeros((),dtype=torch.float64,device=self.device)
        for x,y in self.chunks(scope):
            u=(x+b)/t
            all_lse=torch.logsumexp(u,1)
            positive=u.masked_fill(~y,-torch.inf)
            total += (all_lse-torch.logsumexp(positive,1)).sum(dtype=torch.float64)
            p=torch.softmax(u,1)
            if gradient:
                g=p-torch.softmax(positive,1)
                gb += g.sum(0,dtype=torch.float64)/t
                gt -= (g*u).sum(dtype=torch.float64)
            else:
                q=y.float()/y.sum(1,keepdim=True)
                br += (p-q).square().sum(dtype=torch.float64)
            n+=len(x)
        if not n:raise ValueError("Empty calibration scope")
        result=dict(positive_set_nll=float(total/n),rows=n)
        if not gradient:result["brier"]=float(br/n)
        return result,(float(gt/n),gb.cpu().numpy()/n)


def fit_event_calibration(logits_path, labels_path, groups, device):
    rows=FullRows(logits_path,labels_path,groups,device)
    zero=np.zeros(rows.c,dtype=np.float64)
    history=[]
    def fit(family,scope):
        learn=family=="temperature_and_bias"
        ids=np.flatnonzero(rows.active(scope)) if learn else np.array([],dtype=int)
        evaluations=0;began=time.monotonic()
        def objective(theta):
            nonlocal evaluations
            b=zero.copy()
            if learn:b[ids]=theta[1:]
            stats,(gt,gb)=rows.statistics(theta[0],b,scope,gradient=True)
            loss=stats["positive_set_nll"]+PENALTY*np.square(b).mean()
            grad=np.r_[gt,gb[ids]+2*PENALTY*b[ids]/rows.c] if learn else np.array([gt])
            evaluations+=1
            if evaluations==1 or evaluations%10==0:
                print(f"{scope}/{family}: pass {evaluations}, NLL={stats['positive_set_nll']:.7f}, T={math.exp(theta[0]):.5f}",flush=True)
            return loss,grad
        result=minimize(objective,np.zeros(1+len(ids)),jac=True,method="L-BFGS-B",
                        bounds=[(math.log(.15),math.log(6.))]+[(-4.,4.)]*len(ids),
                        options=dict(maxiter=80,maxfun=120,ftol=1e-9,gtol=3e-6,maxls=15))
        b=zero.copy()
        if learn:b[ids]=result.x[1:]
        candidate=dict(family=family,temperature=math.exp(float(result.x[0])),bias=b.tolist())
        history.append(dict(scope=scope,family=family,objective=float(result.fun),
                            optimizer="bounded full-row analytic-gradient L-BFGS-B",iterations=int(result.nit),
                            full_row_passes=evaluations,converged=bool(result.success),message=str(result.message),
                            elapsed_seconds=time.monotonic()-began,active_classes=ids.tolist()))
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():raise ValueError("Invalid calibration optimization")
        return candidate
    raw=dict(family="raw",temperature=1.,bias=zero.tolist())
    candidates=[raw,fit("temperature_only","fit"),fit("temperature_and_bias","fit")]
    trials=[]
    for candidate in candidates:
        stats,_=rows.statistics(math.log(candidate["temperature"]),candidate["bias"],"selection")
        trials.append(dict(**candidate,selection_nll=stats["positive_set_nll"],selection_brier=stats["brier"]))
    best=min(trials,key=lambda x:x["selection_nll"])
    # Prespecified numerical tie tolerance prefers identity; test is never read.
    if best["selection_nll"] >= trials[0]["selection_nll"]-1e-6:best=trials[0]
    final=raw if best["family"]=="raw" else fit(best["family"],"all")
    rawstats,_=rows.statistics(0.,zero,"all")
    fittedstats,_=rows.statistics(math.log(final["temperature"]),final["bias"],"all")
    guard=fittedstats["positive_set_nll"]<=rawstats["positive_set_nll"]+1e-7
    fitted_temperature=final["temperature"]
    if not guard:final=raw
    result=dict(protocol=PROTOCOL,**final,calibration_rows=rows.n,
        selected_family_before_full_refit_guard=best["family"],full_refit_nll_guard_passed=guard,
        fitted_temperature=fitted_temperature,fitted_temperature_accepted=final["family"]!="raw",
        objective="V5 softmax positive-set negative log likelihood + 0.002 mean-square class bias",
        selection="minimum positive-set NLL on patient/record-disjoint 20% inside external 90%; raw wins improvements <=1e-6; selected family refitted on every external 90% row; full-refit NLL guard",
        selection_split=rows.split,selection_trials=trials,optimization_history=history,
        raw_calibration=rawstats,fitted_calibration=fittedstats,temperature_bounds=[.15,6.],
        bias_bounds=[-4.,4.],bias_penalty=PENALTY,storage=rows.storage,
        all_calibration_rows_used=True,test_labels_used_for_fit_or_selection=False,
        row_logit_shift_invariant=True,model_weights_updated=False)
    del rows
    if torch.device(device).type=="cuda":torch.cuda.empty_cache()
    return result
