#!/usr/bin/env python3
"""Prepare observed history dependence and training-only bedside-score risks."""
import argparse, json, sys, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'outputs/maomao_manuscript_figures_20260929/source_data'
SOURCE = ROOT/'outputs/maomao_plot_sources'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()

def shap_dependence():
    from maomao.data.event_sequence import EventSequenceDataset
    data=ROOT/'data/perioperative_event_sequences_v5_richctx_static7'
    d=EventSequenceDataset(data,256,128,dynamic_windows=False)
    plan_path=ROOT/'outputs/maomao_family_shap_20260929/private_source_window_query_plan.npy'
    plan=np.load(plan_path)
    meta=json.loads((SOURCE/'shap/event_sequence_meta.json').read_text())
    names=meta['outcome_vocabulary']; families=meta['outcome_family_vocabulary']
    death=210+families.index('death'); n=len(plan)
    count=np.zeros((n,210),np.int32); phi=np.zeros((n,210),np.float32)
    last_value=np.full((n,210),np.nan,np.float32); base=np.zeros(n); full=np.zeros(n)
    for i,(w,q) in enumerate(plan):
        ep=int(d.window_admission[w]);lo=int(d.ptr[ep]);end=lo+int(d.window_start[w])+int(q)+1
        raw=np.asarray(d.outcome[lo:end]);valid=np.flatnonzero(raw>=0)
        mapped=d.outcome_remap[raw[valid]];valid=valid[mapped>=0];mapped=mapped[mapped>=0]
        count[i]=np.bincount(mapped,minlength=210)
        hv=np.asarray(d.has_value[lo:end]);values=np.asarray(d.value[lo:end])
        for j,ev in zip(valid,mapped):
            if hv[j] and np.isfinite(values[j]):last_value[i,ev]=values[j]
        p=SOURCE/f'shap/patients/case_{i:05d}.npz';a=np.load(p)
        np.add.at(phi[i],a['input_group_ids']//3,a['shap_values'][:,death])
        base[i]=a['baseline_log_family_probability'][death];full[i]=a['full_log_family_probability'][death]
        if not np.isclose(base[i]+phi[i].sum(dtype=np.float64),full[i],atol=1e-4):raise RuntimeError(f'Case {i} attribution additivity failed')
        if i%1000==0:print('SHAP observed-history extraction',i,n,flush=True)
    OUT.mkdir(exist_ok=True,parents=True)
    np.savez_compressed(OUT/'shap_dependence_observed_history.npz',event_names=np.array(names),
        event_count=count,last_observed_value=last_value,event_shap_to_log_death=phi,
        baseline_log_death=base,full_log_death=full)
    ranking=pd.DataFrame({'event':names,'patients_present':(count>0).sum(0),'patients_with_value':np.isfinite(last_value).sum(0),
        'unique_counts':[len(np.unique(count[:,i])) for i in range(210)],'mean_abs_shap':np.abs(phi).mean(0)})
    ranking.sort_values('mean_abs_shap',ascending=False).to_csv(OUT/'shap_dependence_selection_candidates.csv',index=False)
    proof=dict(complete=True,patients=n,selection='Original one seeded query per each of 9,989 validation patients; every case used',
        target='log next-event original death-family probability',source_plan_sha256=digest(plan_path),
        observed_history='All source tokens up to the original causal query, including pre-window history; counts are untransformed',
        value_definition='Latest observed source value.bin measurement for the event before the causal query; event names ending normalized denote state recovery, not numerical standardization',
        shap_definition='Sum the saved event-identity × lag Partition SHAP over all three lags for each event',
        dependence_interpretation='Observed SHAP dependence colored by a second observed history variable; these are not separately estimated SHAP interaction values or causal effects',
        model_rerun=False,source_protocol_sha256=digest(SOURCE/'shap/protocol.json'))
    (OUT/'shap_dependence_preparation.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('SHAP dependence prepared',flush=True)

ENDPOINTS=[('hypotension_60m',1,'severe_map_hypotension'),('hypotension_6h',6,'severe_map_hypotension'),
 ('pressor_1h',1,'vasopressor_start'),('pressor_6h',6,'vasopressor_start'),('icu_24h',24,'icu_transfer'),
 ('crrt_24h',24,'crrt_start'),('ventilation_24h',24,'ventilation_start'),('rbc_6h',6,'rbc_transfusion'),
 ('rbc_24h',24,'rbc_transfusion'),('troponin_elevation_24h',24,'troponin_i_elevation')]

def clinical_risks():
    from maomao.data.event_sequence import EventSequenceDataset
    from scripts.diagnostics import compare_clinical_scores_expanded as c
    root=ROOT/'outputs/classic_score_comparison/independent_15pct_test'
    private=ROOT/'outputs/maomao_manuscript_figures_20260929/private_risk_training_inputs'
    private.mkdir(exist_ok=True,parents=True)
    dataset=EventSequenceDataset(root/'development',256,128,dynamic_windows=False)
    admissions=pd.read_csv(root/'development/admissions.csv',dtype={'subject_id':str})
    splits=np.load(root/'maomao_fresh/patient_validation_split.npz')
    train=set(splits['train_patients'].tolist());test=set(pd.read_csv(root/'test/admissions.csv',usecols=['subject_id'],dtype=str).subject_id)
    if train & test:raise RuntimeError('Training/test patient overlap')
    base=admissions[admissions.subject_id.isin(train)].copy();base['sequence_id']=base.index
    final=private/'training_scores_and_labels.pkl'
    if final.exists():base=pd.read_pickle(final)
    else:
        stages=[('sequence',lambda:c.collect_sequence_baselines(dataset)),
         ('or',lambda:c.collect_or_inputs(base,ROOT/'data/train_data/vitals.csv.gz')),
         ('ward',lambda:c.collect_preop_vitals(base,ROOT/'data/train_data/ward_vitals.csv.gz')),
         ('labs',lambda:c.collect_preop_labs(base,ROOT/'data/train_data/labs.csv.gz')),
         ('history',lambda:c.collect_history(base,ROOT/'data/train_data/diagnosis.csv.gz',ROOT/'data/train_data/medications.csv.gz'))]
        for key,fn in stages:
            cache=private/f'{key}.pkl';print('Training score inputs',key,flush=True)
            if cache.exists():part=pd.read_pickle(cache)
            else:part=fn();part.to_pickle(cache)
            base=base.merge(part,on='sequence_id' if key=='sequence' else 'op_id',how='left',validate='one_to_one')
        base['preop_spo2']=pd.to_numeric(base.preop_spo2,errors='coerce').fillna(base.baseline_spo2)
        base['duration_min']=pd.to_numeric(base.opend_time,errors='coerce')-pd.to_numeric(base.opstart_time,errors='coerce')
        base=c.derive_scores(base)
        labels=[];surgery=int(dataset.meta['token_vocabulary']['event:surgery_end'])
        for ep in base.sequence_id:
            lo,hi=int(dataset.ptr[ep]),int(dataset.ptr[ep+1]);tokens=np.asarray(dataset.token_id[lo:hi]);marks=np.flatnonzero(tokens==surgery)
            row={'sequence_id':int(ep)}
            if len(marks):
                times=np.asarray(dataset.time_min[lo:hi]);landmark=times[marks[-1]];raw=np.asarray(dataset.outcome[lo:hi]);mapped=np.full(len(raw),-1);ok=raw>=0;mapped[ok]=dataset.outcome_remap[raw[ok]]
                for endpoint,hours,event in ENDPOINTS:
                    ev=dataset.meta['outcome_vocabulary'].index(event);positive=np.any((times>landmark)&(times<=landmark+hours*60)&(mapped==ev))
                    row['label_'+endpoint]=1 if positive else (0 if times[-1]-landmark>=hours*60 else np.nan)
            labels.append(row)
        base=base.merge(pd.DataFrame(labels),on='sequence_id',how='left',validate='one_to_one');base.to_pickle(final)
    if not set(base.subject_id.astype(str))<=train:raise RuntimeError('Calibration source is not training-only')
    prediction_cache=private/'maomao_training_landmark_predictions.pkl'
    if not prediction_cache.exists():raise RuntimeError('Run prepare_clinical_maomao_calibration_inputs.py first')
    cache_proof=json.loads(prediction_cache.with_suffix('.json').read_text())
    if not cache_proof['complete'] or cache_proof['checkpoint_sha256']!=digest(root/'maomao_fresh/best_model.pt'):raise RuntimeError('Stale frozen-model calibration cache')
    base=base.merge(pd.read_pickle(prediction_cache),on='sequence_id',how='inner',validate='one_to_one')
    frame=pd.read_csv(SOURCE/'clinical_score_predictions_no_identifiers.csv')
    dictionary=json.loads((OUT/'clinical_curve_dictionary.json').read_text());fits=[];vectors={};calibration=[];dca=[]
    for endpoint,cohort in dictionary.items():
        specs=cohort['series'];columns=['label_'+endpoint]+[s['score_column'] for s in specs]
        keep=frame[columns].notna().all(1)&np.isfinite(frame[columns].to_numpy(float)).all(1);part=frame.loc[keep];y=part['label_'+endpoint].to_numpy(float)
        if len(part)!=cohort['complete_episodes']:raise RuntimeError('Sealed common-case cohort changed')
        # Fit every comparator on the SAME training complete-case intersection.
        # MAOMAO receives a monotone Platt mapping of its focal-loss logit. The
        # uncalibrated sigmoid is preserved separately, never overwritten.
        train_common=base[columns+['subject_id']].replace([np.inf,-np.inf],np.nan).dropna()
        for spec in specs:
            name=spec['name'];col=spec['score_column'];direction=spec['risk_direction'];prefix=spec['prefix']
            tr=train_common;x=tr[col].to_numpy(float)*direction;yy=tr['label_'+endpoint].to_numpy(float)
            test_x=part[col].to_numpy(float)*direction
            if name=='MAOMAO':
                x=np.log(np.clip(x,1e-7,1-1e-7)/np.clip(1-x,1e-7,1-1e-7))
                test_x=np.log(np.clip(test_x,1e-7,1-1e-7)/np.clip(1-test_x,1e-7,1-1e-7))
            if len(np.unique(yy))!=2:raise RuntimeError(f'{endpoint}/{name} has no two-class training evidence')
            mean=float(x.mean());scale=max(float(x.std()),1e-8);xx=(x-mean)/scale
            def objective(z):
                logits=z[0]+z[1]*xx;pred=expit(logits);pen=1e-4*z[1]**2
                return np.mean(np.logaddexp(0,logits)-yy*logits)+pen,np.array([(pred-yy).mean(),np.mean((pred-yy)*xx)+2e-4*z[1]])
            opt=minimize(objective,[np.log(yy.mean()/(1-yy.mean())),.1],jac=True,method='L-BFGS-B',bounds=[(None,None),(0,None)],options={'maxiter':500,'ftol':1e-12})
            if not opt.success:raise RuntimeError(str(opt.message))
            p=expit(opt.x[0]+opt.x[1]*(test_x-mean)/scale)
            fit=dict(mapping='Training-only monotone logistic risk mapping with fixed risk direction',input_transform='logit' if name=='MAOMAO' else 'identity',train_rows=len(tr),train_patients=int(tr.subject_id.nunique()),train_events=int(yy.sum()),mean=mean,scale=scale,intercept=float(opt.x[0]),slope=float(opt.x[1]),optimizer_success=True,common_training_cohort=True)
            if np.any((p<0)|(p>1)) or not np.isfinite(p).all():raise RuntimeError('Invalid probabilities')
            vectors[prefix+'__risk']=p;vectors[prefix+'__target']=y
            fits.append(dict(endpoint=endpoint,model=name,test_rows=len(y),test_events=int(y.sum()),**fit))
            if name=='MAOMAO':vectors[prefix+'__raw_risk']=part[col].to_numpy(float)
            # Adaptive equal-count bins; tied risks remain together.
            quantiles=np.unique(np.quantile(p,np.linspace(0,1,11)));bin_id=np.searchsorted(quantiles[1:-1],p,side='right')
            for b in np.unique(bin_id):
                hit=bin_id==b;nn=int(hit.sum());obs=float(y[hit].mean());se=np.sqrt(obs*(1-obs)/nn)
                calibration.append(dict(endpoint=endpoint,model=name,bin=int(b),n=nn,events=int(y[hit].sum()),mean_risk=float(p[hit].mean()),observed_fraction=obs,ci_lower=max(0,obs-1.96*se),ci_upper=min(1,obs+1.96*se)))
            # Full threshold curve is preserved; display range is chosen later from risk support.
            threshold=np.unique(np.r_[np.geomspace(1e-6,.1,200),np.linspace(.1,.99,200)])
            for t in threshold:
                positive=p>=t;tp=int(np.sum(positive&(y==1)));fp=int(np.sum(positive&(y==0)))
                dca.append(dict(endpoint=endpoint,model=name,threshold=t,net_benefit=(tp-fp*t/(1-t))/len(y),tp=tp,fp=fp,n=len(y)))
        print('Prepared sealed clinical probabilities',endpoint,cohort['complete_episodes'],flush=True)
    np.savez_compressed(OUT/'clinical_risk_predictions.npz',**vectors)
    pd.DataFrame(calibration).to_csv(OUT/'clinical_calibration_curves.csv',index=False);pd.DataFrame(dca).to_csv(OUT/'clinical_dca_curves.csv',index=False)
    proof=dict(complete=True,training_patient_ids=len(train),training_source_patients=int(base.subject_id.nunique()),test_patient_overlap=0,
        fitted_on_sealed_test=False,internal_validation_patients_used_for_fitting=False,models_refitted=False,
        scope='Fixed original clinical-score 85:15 test; all score-risk mappings INCLUDING MAOMAO fit on the same endpoint-specific complete cases in true MAOMAO training patients within development only',
        protocol='common_training_monotone_risk_mapping_v2',maomao_calibrated=True,
        maomao_raw_interpretation='Uncalibrated sigmoid of weighted focal-loss trajectory logits, alpha_pos=0.75, alpha_neg=0.25, gamma=2; not a validated absolute event probability',
        maomao_training_inference=cache_proof,
        warning='These adapted score risks are newly fitted mappings, not published bedside-score probability equations; DCA is exploratory, not a demonstrated clinical utility gain',
        regularization=1e-4,cohort_policy='The same original endpoint common complete cases as ROC/PR',source_prediction_sha256=digest(SOURCE/'clinical_score_predictions_no_identifiers.csv'),
        training_split_sha256=digest(root/'maomao_fresh/patient_validation_split.npz'),score_mappings=fits)
    (OUT/'clinical_risk_preparation.json').write_text(json.dumps(proof,indent=2)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--clinical',action='store_true');ap.add_argument('--shap',action='store_true');args=ap.parse_args()
    if args.clinical:clinical_risks()
    if args.shap:shap_dependence()
