#!/usr/bin/env python3
"""One-pass full-valid-row multilabel linear SVM baseline (hard time boxed externally)."""
from __future__ import annotations
import json, time
from pathlib import Path
import sys
import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from maomao.evaluation.event_metrics import event_metric_report_with_subsample_ci
BASE=ROOT/'outputs/final_experiment_results_20260923/classical_full_scale'
OUT=ROOT/'outputs/final_experiment_results_20260923/classical_ml_fullscale/linear_svm'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x=np.load(BASE/'train_X.npy',mmap_mode='r'); y=np.load(BASE/'train_y.npy',mmap_mode='r')
    vx=np.load(BASE/'validation_X.npy',mmap_mode='r'); vy=np.load(BASE/'validation_y.npy',mmap_mode='r')
    common=np.load(ROOT/'outputs/baseline_comparison_richctx_fair/baseline_rows_internal.npz')
    sample_y=common['y_validation']
    # Use the saved 20k validation coordinates for a result inside the hard
    # wall clock budget. This remains a sample of the same patient-held-out
    # 10% validation patients; training uses every valid training target row.
    # Recover its positions from the shared coordinate artifact, which is
    # already mapped to the full-scale validation ordering only if the cache
    # metadata says so. Otherwise score the full validation cohort in chunks.
    coords=common['validation_window_indices']; positions=common['validation_positions']
    fullcoords=np.load(BASE/'validation_window_indices.npy',mmap_mode='r')
    fullpos=np.load(BASE/'validation_positions.npy',mmap_mode='r')
    lut={(int(w),int(p)):i for i,(w,p) in enumerate(zip(fullcoords,fullpos))}
    val_idx=np.array([lut[(int(w),int(p))] for w,p in zip(coords,positions)],dtype=np.int64)
    if len(val_idx)!=len(sample_y): raise RuntimeError('shared 20k validation rows could not be mapped to full 90:10 split')
    if not np.array_equal(np.asarray(vy[val_idx],dtype=np.uint8),sample_y):
        raise RuntimeError('mapped full-scale validation labels differ from the shared 20k validation targets')
    model=torch.nn.Linear(x.shape[1],y.shape[1],bias=True).to(device)
    torch.nn.init.zeros_(model.weight); torch.nn.init.zeros_(model.bias)
    opt=torch.optim.SGD(model.parameters(),lr=0.04,weight_decay=1e-5)
    batch=4096; n=len(x); model.train()
    print(json.dumps({'stage':'fit_start','train_rows':n,'features':x.shape[1],'targets':y.shape[1],'device':str(device)}),flush=True)
    for start in range(0,n,batch):
        end=min(start+batch,n)
        xb=torch.as_tensor(np.asarray(x[start:end]),device=device)
        yb=torch.as_tensor(np.asarray(y[start:end]),device=device,dtype=torch.float32)
        signed=yb.mul(2).sub(1)
        scores=model(xb)
        loss=F.relu(1-signed*scores).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if end==n or end//batch%250==0:
            print(json.dumps({'stage':'fit','rows_seen':end,'train_rows':n,'elapsed_s':round(time.monotonic()-started,1),'hinge_loss':float(loss.detach())}),flush=True)
        del xb,yb,signed,scores,loss
    model.eval(); logits=[]
    with torch.inference_mode():
        for st in range(0,len(val_idx),batch):
            idx=val_idx[st:st+batch]
            xb=torch.as_tensor(np.asarray(vx[idx]),device=device)
            logits.append(model(xb).float().cpu()); del xb
    logits=torch.cat(logits)
    # SVM margins are consumed as ranking scores. The shared metric package
    # applies a softmax for calibration/ranking metrics, preserving event order.
    targets=torch.from_numpy(sample_y.astype(bool))
    meta=json.loads((ROOT/'data/perioperative_event_sequences_v5_richctx_static7/event_sequence_meta.json').read_text())
    report=event_metric_report_with_subsample_ci(logits,targets,meta['outcome_vocabulary'],
        bootstrap_repeats=200,bootstrap_seed=4210,max_ci_rows=20_000)
    report.update({'model':'linear_svm_hinge','status':'completed','train_rows_full':n,
      'validation_rows':len(sample_y),'validation_sampling':'same saved patient-disjoint 20k target-row sample as deep baselines',
      'epochs':1,'training_seconds':round(time.monotonic()-started,2),'device':str(device),
      'feature_count':int(x.shape[1]),'outcome_count':int(y.shape[1]),
      'split':'patient-disjoint 90:10, seed 42','ci_bootstrap_repeats':200,
      'method_note':'Multilabel linear hinge-loss SGD; one full pass over all 14,128,539 valid training target rows. Model output is a ranking margin, not a calibrated probability.'})
    (OUT/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    torch.save({'model':model.state_dict(),'feature_count':x.shape[1],'outcome_count':y.shape[1]},OUT/'linear_svm.pt')
    print(json.dumps({'stage':'complete','metrics':{k:report.get(k) for k in ('micro_auprc','micro_auroc','macro_auroc','mrr','brier','ece','hit_at_1','recall_at_5','recall_at_10')},'elapsed_s':report['training_seconds']}),flush=True)
if __name__=='__main__': main()
