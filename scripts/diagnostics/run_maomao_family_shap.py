#!/usr/bin/env python3
"""Partition SHAP for all eligible held-out INSPIRE patients, frozen MAOMAO.

The game masks clinical event contents and corresponding prewindow family counts,
conditional on the observed time grid, intensity, phase, static and other context.
One prespecified seeded eligible window per held-out patient; final valid query.
No model fitting, label selection, model rank manipulation or synthetic SHAP.
"""
import sys,json,time,fcntl,os,argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import shap
from scipy.cluster.hierarchy import linkage

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from maomao.data.event_sequence import EventSequenceDataset,collate_event_sequences
from scripts.diagnostics.evaluate_external_validation import build_model
from scripts.diagnostics.uniform_result_scope import sha256
from scripts.diagnostics.score_uniform_external_five import save_json,stamp
from scripts.diagnostics.maomao_prediction_only import omit_unused_heads

BASE=ROOT/'outputs/final_experiment_results_20260923'
DATA=ROOT/'data/perioperative_event_sequences_v5_richctx_static7'
OUT=ROOT/'outputs/maomao_family_shap_20260929'
PLOT=ROOT/'outputs/maomao_plot_sources/shap'
CHECKPOINT=BASE/'full_maomao_reference/best_model.pt'
PROTOCOL='maomao_family_lag_partition_shap_conditional_content_v2_float32_probability'
LAGS=('recent_lt_2h','intermediate_2_to_24h','remote_ge_24h')

def read(p):return json.loads(p.read_text())
def lagbin(lag):return np.where(lag<2,0,np.where(lag<24,1,2)).astype(int)

def plan(dataset):
    w=np.load(BASE/'classical_full_scale/validation_window_indices.npy',mmap_mode='r')
    pos=np.load(BASE/'classical_full_scale/validation_positions.npy',mmap_mode='r')
    uw,first=np.unique(w,return_index=True)
    end=np.r_[first[1:],len(w)]
    queries={int(win):int(pos[hi-1]) for win,hi in zip(uw,end)}
    admissions=pd.read_csv(DATA/'admissions.csv',dtype={'subject_id':str})
    records=pd.DataFrame(dict(window=uw,subject=admissions.subject_id.to_numpy()[dataset.window_admission[uw]]))
    rng=np.random.default_rng(42);chosen=[]
    for _,g in records.groupby('subject',sort=True):
        win=int(rng.choice(g.window.to_numpy()));chosen.append((win,queries[win]))
    return np.array(chosen,dtype=np.int64)

def prepare(dataset,window,query):
    sample=dataset[int(window)]
    # Causal prefix; labels and target wait time are never model inputs.
    for key in ('token_id','time_min','gap_min','value','has_value','token_kind','phase_id','observation_features',
                'target_set','target_dt_hours','loss_mask','time_mask','trajectory_target','trajectory_mask'):
        sample[key]=sample[key][:int(query)+1]
    ai=int(dataset.window_admission[window]);start=int(dataset.window_start[window]);lo=int(dataset.ptr[ai])
    end=start+int(query)+1
    outcomes=np.asarray(dataset.outcome[lo:lo+end],dtype=int)
    times=np.asarray(dataset.time_min[lo:lo+end],dtype=float)
    valid=np.flatnonzero(outcomes>=0)
    # Stored vocabulary contains input-only stop/context classes. -1 remaps
    # must never index the death class by NumPy negative indexing.
    valid=valid[dataset.outcome_remap[outcomes[valid]]>=0]
    fam=dataset.outcome_family_ids[dataset.outcome_remap[outcomes[valid]]]
    ages=(float(times[-1])-times[valid])/60
    groups=dataset.outcome_remap[outcomes[valid]]*3+lagbin(ages)
    active=np.unique(groups)
    if not len(active):return None
    local=np.full(query+1,-1,dtype=int)
    inside=valid>=start
    local[valid[inside]-start]=np.searchsorted(active,groups[inside])
    history=np.zeros((len(active),dataset.num_event_families),dtype=np.float32)
    for g,f in zip(groups[~inside],fam[~inside]):history[np.searchsorted(active,g),f]+=1
    # Full coalition exactly restores original log1p prewindow counts.
    if not np.allclose(np.log1p(history.sum(0)),sample['history_family_counts'].numpy(),atol=1e-6):
        raise RuntimeError('Prewindow family coalition counts do not restore original input')
    return sample,active,local,history

def game(model,sample,local,history,member,cache_attention=False,inputs_only=True):
    batch=collate_event_sequences([sample]);device=next(model.parameters()).device
    input_keys=('token_id','token_kind','value','has_value','time_min','gap_min','static',
                'observation_features','phase_id','history_family_counts','attention_mask')
    fixed={k:v.to(device) for k,v in batch.items() if torch.is_tensor(v) and (not inputs_only or k in input_keys)}
    local_t=torch.tensor(local,device=device)
    history_t=torch.tensor(history,device=device)
    original_attention=model._attention_mask
    cached_attention=None
    if cache_attention:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            cached_attention=original_attention(fixed['time_min'].expand(32,*fixed['time_min'].shape[1:]).clone(),True)
    def f(coalitions):
        results=[]
        for lo in range(0,len(coalitions),32):
            c=torch.tensor(coalitions[lo:lo+32],dtype=torch.float32,device=device)
            # Fix numerical layout even for the last chunk.
            actual=len(c)
            if actual<32:c=torch.cat([c,c[:1].expand(32-actual,-1)],0)
            b={k:v.expand(32,*v.shape[1:]).clone() for k,v in fixed.items()}
            masked=(local_t[None,:]>=0)&(c[:,local_t.clamp_min(0)]<.5)
            b['token_id'].masked_fill_(masked,3) # existing auxiliary-trained <MASK>
            b['value'].masked_fill_(masked,0.)
            b['has_value'].masked_fill_(masked,0.)
            b['history_family_counts']=torch.log1p(c@history_t)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                if cache_attention:
                    # Observed time grid is fixed in this conditional game;
                    # preserve exact B32 mask values, not an approximation.
                    model._attention_mask=lambda time_min,causal: cached_attention if causal else original_attention(time_min,causal)
                try:logits=model(b).logits[:,-1].float()
                finally:model._attention_mask=original_attention
            # Float32 probability aggregation/log must be outside autocast:
            # matmul within autocast otherwise silently casts family masses.
            with torch.inference_mode(),torch.autocast('cuda',enabled=False):
                p=torch.softmax(logits,1)@member
                scores=torch.cat((torch.log_softmax(logits,1),p.clamp_min(1e-12).log()),1)
            results.append(scores[:actual].cpu().numpy())
        return np.concatenate(results)
    return f

def aggregate():
    files=sorted((PLOT/'patients').glob('*.npz'));meta=read(DATA/'event_sequence_meta.json')
    nfamilies=len(meta['outcome_family_vocabulary']);nevents=len(meta['outcome_vocabulary'])
    ids=np.array(meta['outcome_to_family'])
    sums=np.zeros((3,nfamilies,nfamilies));counts=np.zeros((3,nfamilies),dtype=int)
    event_sums=np.zeros((3,nevents,nevents));event_counts=np.zeros((3,nevents),dtype=int)
    importance=np.zeros((len(files),nfamilies));residual=[];recs=[]
    for j,path in enumerate(files):
        with np.load(path) as a:
            values=a['shap_values'];groups=a['input_group_ids'];residual.append(float(a['additivity_error']))
            family_values={}
            for g,val in zip(groups,values):
                event=int(g)//3;family=ids[event];lag=int(g)%3
                event_sums[lag,event]+=val[:nevents];event_counts[lag,event]+=1
                family_values.setdefault((lag,family),np.zeros(nfamilies))
                family_values[lag,family]+=val[nevents:]
            for (lag,family),val in family_values.items():
                sums[lag,family]+=val;counts[lag,family]+=1
                importance[j,family]+=np.abs(val).mean()
                recs.append(dict(anonymous_case=j,predictor_family=meta['outcome_family_vocabulary'][family],lag_bin=LAGS[lag],mean_abs_log_probability_shap=float(np.abs(val).mean())))
    mean=np.divide(sums,counts[:,:,None],out=np.full_like(sums,np.nan),where=counts[:,:,None]>0)
    np.savez_compressed(PLOT/'family_shap_matrices.npz',mean_log_probability_shap=mean,
        exp_mean_shap=np.exp(mean),patients_with_feature=counts,family_names=np.array(meta['outcome_family_vocabulary']),lag_names=np.array(LAGS))
    np.savez_compressed(PLOT/'family_shap_global_importance.npz',mean_abs_shap_by_case=importance,family_names=np.array(meta['outcome_family_vocabulary']))
    event_mean=np.divide(event_sums,event_counts[:,:,None],out=np.full_like(event_sums,np.nan),where=event_counts[:,:,None]>0)
    np.savez_compressed(PLOT/'event_shap_matrices_by_family.npz',mean_log_probability_shap=event_mean,
        exp_mean_shap=np.exp(event_mean),patients_with_feature=event_counts,event_names=np.array(meta['outcome_vocabulary']),
        family_names=np.array(meta['outcome_family_vocabulary']),outcome_to_family=ids,lag_names=np.array(LAGS))
    pd.DataFrame(recs).to_csv(PLOT/'family_shap_feature_support.csv',index=False)
    status=read(OUT/'status.json')
    save_json(PLOT/'aggregation.json',dict(protocol=PROTOCOL,patients_completed=len(files),patients_eligible=status['eligible_patients'],
        complete=len(files)==status['eligible_patients'],max_additivity_error=max(residual) if residual else None,
        minimum_feature_support_for_display=6,rare_cells_displayed_as_missing=True,
        family_aggregation='Sum event-group SHAP within family and lag per patient, then mean over patients with that family/lag.',
        point_definition='Average grouped Partition SHAP on log next-event family probability, conditional on time grid and other context; exp(mean) is a probability contribution fold, not a causal/hazard ratio.'))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--lane-id',type=int,default=0);parser.add_argument('--lanes',type=int,default=1);parser.add_argument('--initialize',action='store_true');args=parser.parse_args()
    if not 0<=args.lane_id<args.lanes:raise ValueError('Invalid lane')
    status_path=OUT/(f'lane_{args.lane_id}_status.json' if args.lanes>1 else 'status.json')
    OUT.mkdir(exist_ok=True);PLOT.mkdir(parents=True,exist_ok=True);(PLOT/'patients').mkdir(exist_ok=True)
    torch.set_num_threads(2 if args.lanes>1 else 4);torch.manual_seed(42);np.random.seed(42)
    with (OUT/f'worker_lane_{args.lane_id}.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        d=EventSequenceDataset(DATA,256,128,dynamic_windows=False)
        choices=plan(d)
        plan_path=OUT/'private_source_window_query_plan.npy'
        if plan_path.exists():
            if not np.array_equal(np.load(plan_path),choices):raise RuntimeError('Existing private plan differs')
        else:np.save(plan_path,choices)
        with np.load(BASE/'full_maomao_reference/patient_validation_split.npz') as a:total=len(a['validation_patients'])
        protocol=dict(protocol=PROTOCOL,model_sha256=sha256(CHECKPOINT),dataset_meta_sha256=sha256(DATA/'event_sequence_meta.json'),
            heldout_patients=total,eligible_patients=len(choices),excluded_no_valid_query=total-len(choices),
            selection='One uniformly seeded eligible validation window per patient, then last valid target query; seed 42, no selection by labels or prediction performance.',
            explained_output='All 210 log next-event softmax probabilities, plus log(sum probabilities) for all 63 existing semantic families.',
            inputs='Existing event identity × observed lag (<2h, 2–<24h, >=24h); event tokens and prewindow counts co-masked. Family SHAP is the sum of event attributions within each family/lag.',
            masker='Replace withheld clinical event token with trained <MASK>, zero its value and has_value; recompute prewindow family counts from retained coalitions. Time grid, gaps, intensity, phase, static and other context fixed.',
            explainer='shap.PartitionExplainer; complete-linkage tree on family-ordered event/lag group indices; max_evals=500 per trajectory; all 210 event and 63 family outputs.',
            stored_outcome_mapping='Apply stored 224-to-output 210 remap; negative remaps remain input-only fixed context, not explainable event/family features.',
            seed=42,shap_version=shap.__version__,inference_batch=32,precision='bf16 forward; float32 final softmax/log',
            interpretation='Conditional predictive attribution; not event removal intervention, causal effect, hazard ratio or original Delphi log-rate output.')
        if (PLOT/'protocol.json').exists() and read(PLOT/'protocol.json')!=protocol:raise RuntimeError('Different prior SHAP protocol')
        if not (PLOT/'protocol.json').exists():save_json(PLOT/'protocol.json',protocol)
        if args.initialize:
            save_json(OUT/'status.json',dict(status='initialized',eligible_patients=len(choices),patients_completed=0,updated_utc=stamp()));return
        checkpoint=torch.load(CHECKPOINT,map_location='cuda',weights_only=False)
        model=build_model(d,checkpoint,torch.device('cuda'));model.eval()
        equivalence=read(PLOT/'prediction_only_equivalence.json')
        if not equivalence.get('complete') or equivalence['checkpoint_sha256']!=sha256(CHECKPOINT):
            raise RuntimeError('Frozen output-head pruning equivalence not verified')
        omit_unused_heads(model)
        cache_proof=read(PLOT/'fixed_time_cache_equivalence.json')
        if not cache_proof.get('complete') or cache_proof['checkpoint_sha256']!=sha256(CHECKPOINT):
            raise RuntimeError('Fixed-time inference cache equivalence not verified')
        member=torch.nn.functional.one_hot(torch.tensor(d.outcome_family_ids.astype(int),device='cuda'),d.num_event_families).float()
        started=time.time();done=0;assigned=len(choices[args.lane_id::args.lanes])
        for i,(window,query) in enumerate(choices):
            if i%args.lanes!=args.lane_id:continue
            dest=PLOT/f'patients/case_{i:05d}.npz'
            if dest.exists():done+=1;continue
            save_json(status_path,dict(status='running',pid=os.getpid(),eligible_patients=len(choices),assigned_patients=assigned,patients_completed=done,
                lane_id=args.lane_id,lanes=args.lanes,active_anonymous_case=i,elapsed_seconds=time.time()-started,updated_utc=stamp()))
            prepared=prepare(d,int(window),int(query))
            if prepared is None:
                raise RuntimeError(f'Eligible patient {i} lacks clinical-event feature')
            sample,active,local,history=prepared;predict=game(model,sample,local,history,member,cache_attention=True)
            nf=len(active)
            if nf==1:
                baseline=predict(np.zeros((1,1)))[0];full=predict(np.ones((1,1)))[0];values=(full-baseline)[None,:]
            else:
                coordinates=d.outcome_family_ids[active//3].astype(float)*10000+active
                tree=linkage(coordinates[:,None],method='complete')
                masker=shap.maskers.Partition(np.zeros((1,nf)),clustering=tree)
                explainer=shap.PartitionExplainer(predict,masker,output_names=d.meta['outcome_vocabulary']+d.meta['outcome_family_vocabulary'],seed=42+i)
                explanation=explainer(np.ones((1,nf)),max_evals=500,batch_size=32,silent=True)
                values=explanation.values[0];baseline=explanation.base_values[0];full=predict(np.ones((1,nf)))[0]
            err=float(np.max(np.abs(baseline+values.sum(0)-full)))
            if not np.isfinite(values).all() or err>1e-4:raise RuntimeError(f'SHAP additivity failed {i}: {err}')
            temp=dest.with_suffix('.partial.npz')
            np.savez_compressed(temp,shap_values=values.astype(np.float32),input_group_ids=active.astype(np.int16),
                baseline_log_family_probability=baseline,full_log_family_probability=full,additivity_error=err,
                context_tokens=len(local),conditional_history_counts=history.sum(0),query_time_hours=float(sample['time_min'][-1])/60.)
            temp.replace(dest);done+=1
            if done<=5 or done%25==0:print(f'SHAP {done}/{len(choices)} patients; groups={nf}; elapsed={time.time()-started:.1f}s; additivity={err:.2g}',flush=True)
            if args.lanes==1 and (done==25 or done%250==0):aggregate()
        save_json(status_path,dict(status='completed',pid=os.getpid(),eligible_patients=len(choices),assigned_patients=assigned,patients_completed=done,
            lane_id=args.lane_id,lanes=args.lanes,elapsed_seconds=time.time()-started,updated_utc=stamp()))
        if args.lanes==1:aggregate()

if __name__=='__main__':
    try:main()
    except Exception as e:
        previous=read(OUT/'status.json') if (OUT/'status.json').exists() else {}
        save_json(OUT/'status.json',dict(previous,status='failed',error=repr(e),updated_utc=stamp()));raise
