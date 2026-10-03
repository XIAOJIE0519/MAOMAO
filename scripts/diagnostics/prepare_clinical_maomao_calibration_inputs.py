#!/usr/bin/env python3
"""Frozen-model surgery-end scores on actual development training patients only.

Uses the same window and query selection as extract_landmark_predictions. No
model update, test-set fitting, subsampling, or outcome-dependent selection.
The resulting cache contains identifiers and is deliberately not distributed.
"""
import sys, json, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from maomao.data.event_sequence import EventSequenceDataset, collate_event_sequences
from scripts.diagnostics.evaluate_external_validation import build_model
from scripts.diagnostics.prepare_manuscript_revision_sources import ENDPOINTS, digest

def main():
    torch.set_num_threads(4)
    root=ROOT/'outputs/classic_score_comparison/independent_15pct_test'
    private=ROOT/'outputs/maomao_manuscript_figures_20260929/private_risk_training_inputs'
    dest=private/'maomao_training_landmark_predictions.pkl'
    checkpoint_path=root/'maomao_fresh/best_model.pt'
    if dest.exists():
        proof=json.loads(dest.with_suffix('.json').read_text())
        assert proof['checkpoint_sha256']==digest(checkpoint_path)
        print('Using complete frozen training landmark cache',proof,flush=True);return
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    d=EventSequenceDataset(root/'development',256,128,dynamic_windows=False)
    admissions=pd.read_csv(d.path/'admissions.csv',dtype={'subject_id':str})
    train=set(np.load(root/'maomao_fresh/patient_validation_split.npz')['train_patients'].astype(str))
    test=set(pd.read_csv(root/'test/admissions.csv',usecols=['subject_id'],dtype=str).subject_id)
    assert not train & test
    window_by_ep=defaultdict(list)
    for wi,ep in enumerate(d.window_admission):window_by_ep[int(ep)].append(wi)
    by_window=defaultdict(list);surgery=int(d.meta['token_vocabulary']['event:surgery_end'])
    for ep in admissions.index[admissions.subject_id.isin(train)]:
        lo,hi=int(d.ptr[ep]),int(d.ptr[ep+1]);times=np.asarray(d.time_min[lo:hi])
        marks=np.flatnonzero(np.asarray(d.token_id[lo:hi])==surgery)
        if not len(marks):continue
        landmark=float(times[marks[-1]]);pos=int(np.searchsorted(times,landmark,side='right')-1)
        covering=[w for w in window_by_ep[ep] if d.window_start[w]<=pos<d.window_start[w]+d.window_length[w]]
        if covering:
            wi=min(covering,key=lambda w:d.window_start[w]);by_window[wi].append((int(ep),pos-int(d.window_start[wi])))
    selected=sorted(by_window);batch_size=32;device=torch.device('cuda')
    model=build_model(d,checkpoint,device);del checkpoint
    loader=DataLoader(Subset(d,selected),batch_size=batch_size,shuffle=False,num_workers=0,
                      collate_fn=collate_event_sequences,pin_memory=True)
    endpoint_indices=[(name,d.trajectory_horizons_hours.index(float(hours)),d.meta['outcome_vocabulary'].index(event)) for name,hours,event in ENDPOINTS]
    records=[];started=time.monotonic()
    print('Frozen FP32 inference; training-only windows',len(selected),'batches',len(loader),flush=True)
    with torch.inference_mode():
        for bno,batch in enumerate(loader):
            batch={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
            out=model(batch)
            ws=selected[bno*batch_size:(bno+1)*batch_size]
            for bi,wi in enumerate(ws):
                for ep,pos in by_window[wi]:
                    values=torch.sigmoid(out.trajectory_logits[bi,pos]).float().cpu().numpy()
                    records.append({'sequence_id':ep,**{'maomao_'+name:float(values[h,e]) for name,h,e in endpoint_indices}})
            del out,batch
            if bno%20==0 or bno+1==len(loader):
                print(json.dumps(dict(batch=bno+1,total_batches=len(loader),episodes=len(records),elapsed_seconds=round(time.monotonic()-started,1),gpu_peak_gib=round(torch.cuda.max_memory_allocated()/1024**3,2))),flush=True)
    frame=pd.DataFrame(records);assert not frame.sequence_id.duplicated().any()
    assert set(admissions.loc[frame.sequence_id,'subject_id'])<=train
    tmp=dest.with_suffix('.partial.pkl');frame.to_pickle(tmp);tmp.replace(dest)
    proof=dict(complete=True,checkpoint_sha256=digest(checkpoint_path),training_patients=len(train),
               training_episodes=len(frame),windows=len(selected),test_patient_overlap=0,precision='FP32',
               frozen_model=True,models_refitted=False,selection='All available surgery-end landmarks in actual training patients; earliest covering window; last same-time token, identical to sealed test extraction',
               elapsed_seconds=time.monotonic()-started,cache_sha256=digest(dest))
    dest.with_suffix('.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('Completed training-only calibration inputs',proof,flush=True)
if __name__=='__main__':main()
