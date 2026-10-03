#!/usr/bin/env python3
"""Independently reconstruct all saved SHAP feature groups and summaries."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from maomao.data.event_sequence import EventSequenceDataset
from scripts.diagnostics.uniform_result_scope import sha256
from scripts.diagnostics.maomao_display_family_groups import display_mapping
BASE=ROOT/'outputs/final_experiment_results_20260923'
DATA=ROOT/'data/perioperative_event_sequences_v5_richctx_static7'
OUT=ROOT/'outputs/maomao_plot_sources/shap'
WORK=ROOT/'outputs/maomao_family_shap_20260929'
def read(p):return json.loads(p.read_text())
def main():
    errors=[];protocol=read(OUT/'protocol.json');agg=read(OUT/'aggregation.json')
    d=EventSequenceDataset(DATA,256,128,dynamic_windows=False)
    w=np.load(BASE/'classical_full_scale/validation_window_indices.npy',mmap_mode='r')
    pos=np.load(BASE/'classical_full_scale/validation_positions.npy',mmap_mode='r')
    uw,first=np.unique(w,return_index=True);ends=np.r_[first[1:],len(w)]
    queries=dict(zip(uw.astype(int),np.asarray(pos[ends-1],dtype=int)))
    admissions=pd.read_csv(DATA/'admissions.csv',dtype={'subject_id':str})
    subjects=admissions.subject_id.to_numpy()[d.window_admission[uw]]
    with np.load(BASE/'full_maomao_reference/patient_validation_split.npz') as s:
        train=set(s['train_patients'].astype(str));heldout=set(s['validation_patients'].astype(str))
    if train&heldout:errors.append('Training/holdout patients overlap')
    if set(subjects)!=heldout:errors.append('Eligible query patients differ from heldout patients')
    records=pd.DataFrame(dict(window=uw,subject=subjects));rng=np.random.default_rng(42);choices=[]
    for subject,g in records.groupby('subject',sort=True):
        win=int(rng.choice(g.window.to_numpy()));choices.append((win,queries[win]))
    choices=np.array(choices,dtype=np.int64)
    if not np.array_equal(choices,np.load(WORK/'private_source_window_query_plan.npy')):errors.append('Private seeded plan differs from independent reconstruction')
    if protocol['heldout_patients']!=len(heldout) or protocol['eligible_patients']!=len(choices):errors.append('Protocol patient counts differ')
    files=sorted((OUT/'patients').glob('case_*.npz'))
    if [p.name for p in files]!=[f'case_{i:05d}.npz' for i in range(len(choices))]:errors.append('Missing or unexpected case files')
    if not agg['complete'] or len(files)!=len(choices):raise RuntimeError('SHAP cohort is not complete')
    ids=np.array(d.meta['outcome_to_family']);ne=len(ids);nf=d.num_event_families
    event_sum=np.zeros((3,ne,ne));event_count=np.zeros((3,ne),int)
    family_sum=np.zeros((3,nf,nf));family_count=np.zeros((3,nf),int)
    importance=np.zeros((len(files),nf));death_signed=np.zeros_like(importance);presence=np.zeros_like(importance,bool)
    display,display_names=display_mapping(d.meta['outcome_family_vocabulary']);display=np.array(display)
    coarse_importance=np.zeros((len(files),len(display_names)))
    max_residual=0.;max_normalization=0.;min_groups=999;max_groups=0;history_only_cases=0
    hashes=[];death=d.meta['outcome_family_vocabulary'].index('death')
    for i,(path,(window,query)) in enumerate(zip(files,choices)):
        ai=int(d.window_admission[window]);start=int(d.window_start[window]);lo=int(d.ptr[ai]);stop=lo+start+int(query)+1
        stored=np.asarray(d.outcome[lo:stop],dtype=int);times=np.asarray(d.time_min[lo:stop],dtype=float)
        locations=np.flatnonzero(stored>=0);mapped=d.outcome_remap[stored[locations]]
        keep=mapped>=0;locations=locations[keep];mapped=mapped[keep]
        age=(times[-1]-times[locations])/60.;lag=np.select([age<2,age<24],[0,1],default=2)
        expected_groups=np.unique(3*mapped+lag)
        expected_history=np.bincount(ids[mapped[locations<start]],minlength=nf)
        with np.load(path,allow_pickle=False) as a:
            groups=a['input_group_ids'].astype(int);v=a['shap_values'].astype(float)
            base=a['baseline_log_family_probability'].astype(float);full=a['full_log_family_probability'].astype(float)
            if not np.array_equal(groups,expected_groups):errors.append(f'{path.name}: input groups differ from observed remapped clinical history')
            if not np.array_equal(a['conditional_history_counts'],expected_history):errors.append(f'{path.name}: prewindow counts differ')
            if int(a['context_tokens'])!=int(query)+1 or abs(float(a['query_time_hours'])-times[-1]/60)>1e-6:errors.append(f'{path.name}: context/query mismatch')
            if v.shape!=(len(groups),ne+nf) or not np.isfinite(v).all() or np.any(groups<0) or np.any(groups>=ne*3):errors.append(f'{path.name}: invalid attribution values/groups')
            residual=float(np.abs(base+v.sum(0)-full).max());max_residual=max(max_residual,residual)
            if residual>1e-4:errors.append(f'{path.name}: stored float32 additivity residual {residual}')
            for values in (base,full):
                prob=np.exp(values[:ne]);family_prob=np.bincount(ids,weights=prob,minlength=nf)
                err=max(abs(prob.sum()-1.),float(np.abs(np.exp(values[ne:])-np.maximum(family_prob,1e-12)).max()))
                max_normalization=max(max_normalization,err)
                if err>5e-6:errors.append(f'{path.name}: event/family probability normalization differs')
            by_family={};by_coarse={}
            for group,value in zip(groups,v):
                event=group//3;bin_id=group%3;family=ids[event];coarse=display[family]
                event_sum[bin_id,event]+=value[:ne];event_count[bin_id,event]+=1
                by_family.setdefault((bin_id,family),np.zeros(nf));by_family[(bin_id,family)]+=value[ne:]
                by_coarse.setdefault((bin_id,coarse),np.zeros(nf));by_coarse[(bin_id,coarse)]+=value[ne:]
                death_signed[i,family]+=value[ne+death];presence[i,family]=True
            for (bin_id,family),value in by_family.items():
                family_sum[bin_id,family]+=value;family_count[bin_id,family]+=1;importance[i,family]+=np.abs(value).mean()
            for (bin_id,coarse),value in by_coarse.items():coarse_importance[i,coarse]+=np.abs(value).mean()
            min_groups=min(min_groups,len(groups));max_groups=max(max_groups,len(groups))
            if len(expected_history.nonzero()[0]):history_only_cases+=1
        hashes.append(dict(path=path.relative_to(OUT).as_posix(),sha256=sha256(path)))
        if (i+1)%1000==0:print(f'Verified actual source groups + stored residuals: {i+1}/{len(files)}',flush=True)
    for name,total,count in [('event_shap_matrices_by_family.npz',event_sum,event_count),('family_shap_matrices.npz',family_sum,family_count)]:
        means=np.divide(total,count[:,:,None],out=np.full_like(total,np.nan),where=count[:,:,None]>0)
        with np.load(OUT/name) as a:
            if not np.array_equal(a['patients_with_feature'],count) or not np.allclose(a['mean_log_probability_shap'],means,rtol=1e-10,atol=1e-10,equal_nan=True):errors.append(f'{name}: aggregate differs from all per-patient attributions')
            if not np.allclose(a['exp_mean_shap'],np.exp(means),rtol=1e-10,atol=1e-10,equal_nan=True):errors.append(f'{name}: folds differ')
    with np.load(OUT/'family_shap_global_importance.npz') as a:
        if not np.allclose(a['mean_abs_shap_by_case'],importance,rtol=1e-10,atol=1e-10):errors.append('Fine family importance not reproduced')
    np.savez_compressed(OUT/'clinical_group_shap_importance.npz',mean_abs_shap_by_case=coarse_importance,clinical_group_names=np.array(display_names),
        interpretation='Sum input-event SHAP within each clinical group and lag, then mean absolute over 63 original family outputs, then sum lags. Does not claim SHAP of 17 aggregated output probabilities.')
    if (OUT/'death_family_shap_by_patient.npz').exists():
        with np.load(OUT/'death_family_shap_by_patient.npz') as a:
            if a['signed_shap'].shape==death_signed.shape and not np.allclose(a['signed_shap'],death_signed,rtol=1e-6,atol=1e-6):errors.append('Final death family source differs')
    budget=read(OUT/'budget_sensitivity.json')
    if budget['status']!='completed' or budget['cases']!=25:errors.append('Budget sensitivity evidence missing')
    if protocol['model_sha256']!=sha256(BASE/'full_maomao_reference/best_model.pt') or protocol['dataset_meta_sha256']!=sha256(DATA/'event_sequence_meta.json'):errors.append('Frozen model/data source hash differs')
    result=dict(complete=not errors,errors=errors,heldout_patients=len(heldout),patients_verified=len(files),train_patients=len(train),
        patient_disjoint=True,all_eligible_patients=True,one_seeded_query_per_patient=True,all_query_rows_explained=False,
        causal_observed_input_groups_checked_against_actual_source=True,prewindow_counts_checked_against_actual_source=True,
        max_stored_float32_additivity_residual=max_residual,max_event_family_probability_error=max_normalization,
        input_group_range=[min_groups,max_groups],cases_with_prewindow_history=history_only_cases,
        protocol_sha256=sha256(OUT/'protocol.json'),aggregation_sha256=sha256(OUT/'aggregation.json'),
        budget_sensitivity_max_single_attribution_change=max(x['max_absolute_shap_difference'] for x in budget['results']),
        budget_sensitivity_mean_case_mean_absolute_change=float(np.mean([x['mean_absolute_shap_difference'] for x in budget['results']])),
        finite_budget_approximation=True,convergence_not_claimed=True,files=hashes)
    result['source_code_sha256']={name:sha256(ROOT/name) for name in ('maomao/models/event_maomao.py','maomao/data/event_sequence.py',
        'scripts/diagnostics/run_maomao_family_shap.py','scripts/diagnostics/maomao_prediction_only.py','scripts/diagnostics/verify_maomao_shap.py')}
    (OUT/'numerical_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
    if errors:raise SystemExit(1)
if __name__=='__main__':main()
