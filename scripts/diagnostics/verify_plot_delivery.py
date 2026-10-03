#!/usr/bin/env python3
"""Read-only checks of all-row figure mass, metric identities and revised calibration."""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.diagnostics.build_fullscale_external_rows import SOURCES
from maomao.evaluation.softmax_calibration import PROTOCOL, LEGACY_PROTOCOL, revision_directory
from maomao.evaluation.event_bias_calibration import validate_fit
from scripts.diagnostics.uniform_result_scope import sha256
from maomao.evaluation.rank_statistics import RANK_VERSION
from maomao.evaluation.plot_sources import RETRIEVAL_TIE_PROTOCOL

OUT=ROOT/"outputs/maomao_plot_sources"
EXT=ROOT/"outputs/external_validation_final_maomao_uniform"
REV=revision_directory()


def read(path):return json.loads(path.read_text())


def main():
    errors=[];checks=[];areas=[]
    status=read(REV/"queue_status.json")
    main_count=sum(str(x).startswith('main/') for x in status.get('completed',[])) if PROTOCOL!=LEGACY_PROTOCOL else len(status.get('completed',[]))
    if status.get("status")!="completed" or main_count!=40 or not status.get("promoted"):
        errors.append("calibration revision not completed/promoted")
    expected={}
    base=ROOT/"outputs/final_experiment_results_20260923"
    for name in ("univariate","logistic_regression","xgboost","ann","maomao"):
        expected[f"internal/{name}"]=1563972
    for path in (base/"module_metrics").glob("no_*.json"):
        expected[f"module_ablations/{path.stem}"]=read(path)["event_targets"]
    for path in (ROOT/"outputs/scale_ablations_richctx_20260928/metrics").glob("*.json"):
        if not path.stem.endswith("_time_scales"):expected[f"scale_ablations/{path.stem}"]=read(path)["event_targets"]
    for site in SOURCES:
        manifest=read(base/f"classical_full_scale/external/{site}/manifest.json")
        for model in ("univariate","logistic_regression","xgboost","ann","maomao"):
            result=read(EXT/site/model/"metrics.json");fit=read(EXT/site/model/"calibration_fit.json")
            if result.get("calibration_protocol")!=PROTOCOL or fit.get("protocol")!=PROTOCOL:
                errors.append(f"{site}/{model}: wrong calibration protocol")
            if result.get("rank_metric_definition")!=RANK_VERSION:errors.append(f"{site}/{model}: obsolete rank statistics")
            if fit.get("test_labels_used_for_fit_or_selection") is not False or not fit.get("row_logit_shift_invariant"):
                errors.append(f"{site}/{model}: test use/gauge check failed")
            if fit["calibration_rows"]!=manifest["calibration_target_rows"] or result["test_rows"]!=manifest["test_target_rows"]:
                errors.append(f"{site}/{model}: incomplete calibration/test rows")
            if PROTOCOL!=LEGACY_PROTOCOL:
                try:validate_fit(fit,manifest['calibration_target_rows'],210)
                except AssertionError:errors.append(f'{site}/{model}: V5 calibration-only selection/refit invalid')
                if result.get('bias')!=fit['bias'] or result.get('calibration_family')!=fit['family'] or result['temperature']!=fit['temperature']:
                    errors.append(f'{site}/{model}: saved V5 transform differs')
            else:
                selected=fit["fitted_temperature_accepted"]
                if selected != (fit["fitted_calibration"]["brier"]<fit["raw_calibration"]["brier"]-1e-8):
                    errors.append(f"{site}/{model}: calibration-only selection not followed")
                if result["temperature"]!=(fit["fitted_temperature"] if selected else 1.):
                    errors.append(f"{site}/{model}: wrong adopted temperature")
            expected[f"external/{site}/{model}/after"]=result["test_rows"]
        raw=read(EXT/site/"maomao/metrics_uncalibrated.json");post=read(EXT/site/"maomao/metrics.json")
        if raw["calibrated_reference_sha256"]!=sha256(EXT/site/"maomao/metrics.json"):
            errors.append(f"{site}/maomao: stale raw-post pair")
        from scripts.diagnostics.complete_external_retrieval import KEYS
        if PROTOCOL==LEGACY_PROTOCOL or not np.any(post.get('bias',[0.])):
            for metric in ("mrr","recall_at_5","recall_at_10",*KEYS):
                if abs(raw[metric]-post[metric])>1e-7:errors.append(f"{site}/maomao: zero-bias transform changed within-row rank")
            with np.load(OUT/f'external/{site}/maomao/before/retrieval_rows_no_identifiers.npz') as before, np.load(OUT/f'external/{site}/maomao/after/retrieval_rows_no_identifiers.npz') as after:
                if not np.array_equal(before['hits'],after['hits']) or not np.array_equal(before['top10_prediction_event_ids'],after['top10_prediction_event_ids']):
                    errors.append(f'{site}/maomao: zero-bias transform changed exact retrieval rows')
        expected[f"external/{site}/maomao/before"]=raw["test_rows"]
    for relative,n in expected.items():
        folder=OUT/relative
        if not (folder/"provenance.json").exists():
            errors.append(f"{relative}: no figure provenance");continue
        meta=read(folder/"provenance.json")
        if meta.get("rank_metric_definition")!=RANK_VERSION:errors.append(f"{relative}: obsolete point statistics version")
        if meta.get("retrieval_tie_protocol")!=RETRIEVAL_TIE_PROTOCOL:errors.append(f"{relative}: plot/point class tie ordering differs")
        metric_source=ROOT/meta["source_metric_file"]
        point=read(metric_source)
        if relative.startswith('external/'):
            from scripts.diagnostics.complete_external_retrieval import KEYS,VERSION
            vector=folder/'retrieval_rows_no_identifiers.npz'
            if not vector.exists():errors.append(f'{relative}: full retrieval row vectors missing')
            else:
                with np.load(vector) as a:
                    hit=a['hits'];first=a['first_positive_rank'];last=a['last_positive_rank'];card=a['positive_count']
                    if a['metric_names'].tolist()!=list(KEYS) or hit.shape!=(n,len(KEYS)) or np.any(card<1):errors.append(f'{relative}: retrieval vector shape/definition invalid')
                    if not np.array_equal(hit[:,:3],np.column_stack([first<=k for k in (1,5,10)])) or not np.array_equal(hit[:,3],last<=10):errors.append(f'{relative}: exact retrieval does not match ranks')
                    for i,key in enumerate(KEYS):
                        if abs(hit[:,i].mean()-point.get(key,-1))>2e-7:errors.append(f'{relative}: {key} point not full rows')
                    defs=read(OUT/'external_retrieval_definitions.json')
                    fine=np.array(defs['outcome_to_family']);coarse=np.array(defs['family_to_clinical_group'])[fine]
                    top=a['top10_prediction_event_ids'];packed=a['packed_true_events']
                    site=relative.split('/')[1]
                    ysource=np.load(ROOT/f'outputs/final_experiment_results_20260923/classical_full_scale/external/{site}/test_y.npy',mmap_mode='r')
                    for lo in range(0,n,8192):
                        hi=min(lo+8192,n);true=np.unpackbits(packed[lo:hi],axis=1,count=210,bitorder='little').astype(bool)
                        if not np.array_equal(true,ysource[lo:hi]):errors.append(f'{relative}: packed true rows differ from full sealed source');break
                        order=top[lo:hi].astype(int);rows=np.arange(hi-lo)[:,None]
                        if np.any(order<0) or np.any(order>=210):errors.append(f'{relative}: invalid top-10 event ids');break
                        event_hits=true[rows,order]
                        reconstructed_hits=np.column_stack([event_hits[:,:k].any(1) for k in (1,5,10)]+[event_hits.sum(1)==true.sum(1)])
                        levels=[]
                        for mapping in (fine,coarse):
                            t=np.zeros((len(true),int(mapping.max())+1),dtype=bool)
                            for c in range(210):t[:,mapping[c]]|=true[:,c]
                            values=[]
                            for k in (1,5,10):
                                pred=np.zeros_like(t);pred[rows,mapping[order[:,:k]]]=True
                                values.append((t&pred).any(1))
                            values.append((~t|pred).all(1));levels.append(np.column_stack(values))
                        if not np.array_equal(hit[lo:hi],np.column_stack([reconstructed_hits,*levels])):
                            errors.append(f'{relative}: event/fine/coarse hits not reproduced from every actual top-10 prediction');break
                if point.get('retrieval_protocol')!=VERSION or sha256(vector)!=point.get('retrieval_vector_sha256'):errors.append(f'{relative}: retrieval provenance stale')
        if meta["source_metric_sha256"]!=sha256(metric_source):errors.append(f"{relative}: stale point source fingerprint")
        with np.load(folder/"full_row_curves.npz") as arrays:
            p=arrays["positive_histogram"];negative=arrays["negative_histogram"]
            outcome_names=arrays["outcome_names"].tolist()
            if not np.array_equal((p+negative).sum(1),np.full(len(p),n)):
                errors.append(f"{relative}: ROC/PR histogram mass mismatch")
            for event,ph,nh in [("micro",p.sum(0),negative.sum(0)),*[(name,p[i],negative[i]) for i,name in enumerate(outcome_names)]]:
                ph=ph.astype(np.float64);nh=nh.astype(np.float64);pos=ph.sum();neg=nh.sum()
                outside=float((ph*(nh.cumsum()-nh)).sum())
                within=float((ph*nh).sum())
                if event!="micro" and point["per_event"][event]["support"]!=int(pos):errors.append(f"{relative}/{event}: histogram positive support differs")
                exact_auc=point["micro_auroc"] if event=="micro" else point["per_event"][event]["auroc"]
                if pos and neg and exact_auc is not None and not outside/(pos*neg)-1e-7<=exact_auc<=(outside+within)/(pos*neg)+1e-7:
                    errors.append(f"{relative}/{event}: reported AUROC outside full histogram bounds")
                tp=ph[::-1].cumsum();fp=nh[::-1].cumsum()
                ap=float((ph[::-1]*np.divide(tp,tp+fp,out=np.zeros_like(tp),where=tp+fp>0)).sum()/pos) if pos else None
                areas.append(dict(source_state=relative,event=event,positive_cells=int(pos),negative_cells=int(neg),
                                  binned_tie_aware_auroc=(outside+.5*within)/(pos*neg) if pos and neg else None,
                                  exact_tie_aware_auroc_lower=outside/(pos*neg) if pos and neg else None,
                                  exact_tie_aware_auroc_upper=(outside+within)/(pos*neg) if pos and neg else None,
                                  binned_threshold_grouped_ap=ap,curve_approximation=True))
        rel=pd.read_csv(folder/"reliability.csv")
        ranks=pd.read_csv(folder/"first_positive_rank.csv")
        cardinality=pd.read_csv(folder/"target_cardinality.csv")
        with np.load(folder/"top1_fractional_confusion.npz") as a:
            confusion_sum=float(a["matrix"].sum())
        if rel.rows.sum()!=n or ranks.rows.sum()!=n or cardinality.rows.sum()!=n or abs(confusion_sum-n)>1e-5:
            errors.append(f"{relative}: distribution mass mismatch")
        # Reported retrieval means retain float32 reduction; the integer rank
        # histogram reconstruction is float64. Use a 2e-7 reduction tolerance.
        if abs(rel.sum_any_positive_correct.sum()/n-point["hit_at_1"])>2e-7 or abs((ranks.rows/ranks['rank']).sum()/n-point["mrr"])>2e-7:
            errors.append(f"{relative}: plot ranks/reliability do not reproduce point Hit/MRR")
        if meta.get("rows")!=n or meta.get("all_prediction_rows_used") is not True:
            errors.append(f"{relative}: row/provenance mismatch")
        if abs(meta.get("reported_brier",meta["brier_normalized_target"])-meta["brier_normalized_target"])>5e-6:
            errors.append(f"{relative}: figure probabilities differ from reported point Brier")
        with np.load(folder/"target_row_alignment.npz") as a:
            index=a["target_row_index"]
            if len(index)!=n or np.any(index[1:]<=index[:-1]):errors.append(f"{relative}: target coordinate alignment invalid")
        t=folder/"time_predictions_no_identifiers.npz"
        if t.exists():
            ts=read(folder/"time_provenance.json")
            with np.load(t) as a:
                actual=a["actual_wait_hours"];pred=a["predicted_positive_set_average_hours"];err=a["absolute_error_hours"]
                if len(actual)!=n or not np.allclose(np.abs(pred-actual),err,rtol=0,atol=1e-5):errors.append(f"{relative}: time vector mismatch")
                if not np.isfinite(actual).all() or not np.isfinite(pred).all():errors.append(f"{relative}: nonfinite time")
            if sum(x["rows"] for x in ts["strata"].values())!=n:errors.append(f"{relative}: missing time strata")
        checks.append(dict(path=relative,rows=n,classes=len(p),histogram_mass_verified=True,time_vectors=t.exists(),
                           plot_minus_point_hit_at_1=float(rel.sum_any_positive_correct.sum()/n-point["hit_at_1"]),
                           plot_minus_point_mrr=float((ranks.rows/ranks['rank']).sum()/n-point["mrr"])))
    clinical=read(OUT/"clinical_curves/pair_index.json")
    if len(clinical)!=42:errors.append("clinical curve pair count !=42")
    clinical_csv=pd.read_csv(OUT/"clinical_score_predictions_no_identifiers.csv")
    for column in ("subject_id","hadm_id","op_id","case_id","sequence_id"):
        if column in clinical_csv:errors.append(f"raw clinical identifier exported: {column}")
    if "anonymous_patient_cluster" not in clinical_csv:errors.append("clinical anonymous clustering missing")
    long=pd.read_csv(OUT/"metrics_long.csv")
    if len(long)!=710:errors.append(f"expected 71 states × 10 metrics, found {len(long)}")
    retrieval=pd.read_csv(OUT/'external_retrieval_metrics.csv')
    if len(retrieval)!=576:errors.append(f'expected 48 external states × 12 retrieval metrics, found {len(retrieval)}')
    interval_proof=read(OUT/'external_retrieval_interval_verification.json')
    if not interval_proof.get('complete') or interval_proof.get('states')!=48:errors.append('Full retrieval interval independent reconstruction missing')
    for check in interval_proof.get('checks',[]):
        source=EXT/check['site']/check['model']/('metrics_uncalibrated.json' if check['state']=='before' else 'metrics.json')
        if check['source_metric_sha256']!=sha256(source):errors.append('Retrieval CI proof source changed')
    if len(expected)!=71:errors.append(f"expected 71 curve states, found {len(expected)}")
    for path in OUT.rglob("*"):
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(OUT).parts) and path.suffix in (".pt",".bin"):
            errors.append(f"raw checkpoint/data found in plot sources: {path}")
    result=dict(complete=not errors,errors=errors,curve_states=len(checks),clinical_pairs=len(clinical),checks=checks,
                retrieval_float32_vs_histogram_float64_mean_tolerance=2e-7,
                calibration_protocol=PROTOCOL,full_row_histograms=True,raw_patient_identifiers_in_clinical_csv=False)
    (OUT/"delivery_verification.json").write_text(json.dumps(result,indent=2)+"\n")
    pd.DataFrame(areas).to_csv(OUT/"curve_areas_and_auroc_bounds.csv",index=False)
    print(json.dumps({k:v for k,v in result.items() if k!="checks"},ensure_ascii=False))
    if errors:raise SystemExit(1)


if __name__=="__main__":main()
