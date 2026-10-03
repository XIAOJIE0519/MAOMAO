#!/usr/bin/env python3
"""Full 10% internal-validation MAE stratified by dual-time-scale bins."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from maomao.data.event_sequence import EventSequenceDataset,collate_event_sequences
from maomao.models.event_maomao import EventMAOMAO,dual_timescale_expected_wait
from scripts.diagnostics.run_baseline_comparison import patient_windows

OUT=ROOT/'outputs/final_experiment_results_20260923/model_metrics/maomao_dual_timescale_mae_internal.json'
CHECKPOINT=ROOT/'outputs/final_experiment_results_20260923/full_maomao_reference/best_model.pt'
DATA=ROOT/'data/perioperative_event_sequences_v5_richctx_static7'

def boot_ci(errors: np.ndarray, seed: int=42, repeats: int=200):
    if len(errors)<2:return None
    rng=np.random.default_rng(seed); means=np.empty(repeats,dtype=np.float64)
    for i in range(repeats):
        means[i]=errors[rng.integers(0,len(errors),size=len(errors))].mean()
    lo,hi=np.quantile(means,[.025,.975]);return [float(lo),float(hi)]

def main():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=EventSequenceDataset(DATA,256,128,dynamic_windows=False)
    payload=torch.load(CHECKPOINT,map_location=device,weights_only=False);cfg=payload['args']
    if not cfg.get('dual_timescale_time_head',False):raise RuntimeError('Selected checkpoint does not have a dual-timescale time head')
    _,val_idx,split=patient_windows(ds,.1,42)
    loader=DataLoader(Subset(ds,val_idx),batch_size=32,shuffle=False,num_workers=0,
        pin_memory=device.type=='cuda',collate_fn=collate_event_sequences)
    model=EventMAOMAO(ds.num_tokens,ds.num_outcomes,ds.num_static,
        cfg.get('hidden_dim',384),cfg.get('num_layers',10),cfg.get('num_heads',12),
        cfg.get('ffn_dim',1536),cfg.get('dropout',.1),cfg.get('initial_event_interval_hours',24.),
        cfg.get('decoupled_time_head',False),cfg.get('enhanced_time_encoding',False),
        len(ds.trajectory_horizons_hours),cfg.get('lognormal_time_head',False),
        outcome_family_ids=torch.as_tensor(ds.meta['outcome_to_family'],dtype=torch.long),
        use_family_head=cfg.get('use_family_head',True),
        same_time_block_causal=cfg.get('same_time_block_causal',False),
        relative_time_attention=cfg.get('relative_time_attention',False),
        event_conditioned_time_head=cfg.get('event_conditioned_time_head',False),
        phase_memory=cfg.get('phase_memory',False),clock_phase_context=cfg.get('clock_phase_context',True),
        observation_intensity=cfg.get('observation_intensity',False),
        value_reconstruction=cfg.get('masked_value_loss_weight',0.)>0,
        dual_timescale_time_head=True,fine_time_bins=cfg.get('fine_time_bins',24),
        long_time_bins=cfg.get('long_time_bins',44)).to(device)
    clock_ids=[int(i) for token,i in ds.meta['token_vocabulary'].items() if token=='<CLOCK>' or token.startswith('phase_summary:')]
    model.clock_phase_token_ids=torch.as_tensor(clock_ids,dtype=torch.long,device=device)
    model.load_state_dict(payload['model']);model.eval()
    end_long=2.+.5*int(cfg.get('long_time_bins',44))
    groups={'fine_0_to_2h':[],'long_2h_to_tail_start':[],'tail_from_tail_start':[]}
    with torch.inference_mode():
      for i,batch in enumerate(loader,1):
        batch={k:(v.to(device,non_blocking=True) if torch.is_tensor(v) else v) for k,v in batch.items()}
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=device.type=='cuda'):
          output=model(batch)
        if output.fine_hazard_logits is None:raise RuntimeError('Dual-time-scale hazards were not returned')
        wait_by_event=dual_timescale_expected_wait(output.fine_hazard_logits,output.long_hazard_logits,
                                                   output.tail_mu,output.tail_log_sigma)
        targets=batch['target_set'].float();target_n=targets.sum(-1)
        valid=batch['loss_mask'].bool() & (target_n>0)
        actual=batch['target_dt_hours'].float()
        predicted=(wait_by_event*targets).sum(-1)/target_n.clamp_min(1)
        abs_error=(predicted-actual).abs()
        masks={
          'fine_0_to_2h':valid & (actual<2.),
          'long_2h_to_tail_start':valid & (actual>=2.) & (actual<end_long),
          'tail_from_tail_start':valid & (actual>=end_long),
        }
        for name,mask in masks.items():
          if mask.any():groups[name].append(abs_error[mask].float().cpu().numpy())
        if i%100==0:print(f'eval_batches={i}/{len(loader)}',flush=True)
    names={
      'fine_0_to_2h':f'Fine scale (<2 h; 24 five-minute hazard bins)',
      'long_2h_to_tail_start':f'Long scale (2 to <{end_long:g} h; {int(cfg.get("long_time_bins",44))} half-hour bins)',
      'tail_from_tail_start':f'Long-tail scale (≥{end_long:g} h; log-normal tail)',
    }
    result={'model':'MAOMAO dual-timescale head','checkpoint':str(CHECKPOINT.relative_to(ROOT)),
      'checkpoint_epoch':int(payload.get('epoch',-1)),'split':split,
      'evaluation_scope':'all valid event-target positions in patient-disjoint internal 10% validation split',
      'validation_windows':len(val_idx),'bucket_boundaries_hours':{'fine':[0,2],'long':[2,end_long],'tail':[end_long,None]},
      'time_mae_by_scale':{},'ci_method':'200 bootstrap resamples of event-target rows within each scale; percentile 95% CI',
      'pooled_time_mae_reported':False}
    for idx,(key,label) in enumerate(names.items()):
      errors=np.concatenate(groups[key]) if groups[key] else np.empty(0,dtype=np.float32)
      result['time_mae_by_scale'][key]={'label':label,'event_target_rows':int(len(errors)),
        'mae_hours':float(errors.mean()) if len(errors) else None,
        'mae_hours_95ci':boot_ci(errors,seed=4200+idx) if len(errors) else None}
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
