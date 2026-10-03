#!/usr/bin/env python3
"""Full sealed external comparisons of the saved module/scale checkpoints.

No fitting of model weights. Temperature fitting uses only the frozen external
90% calibration patients. Vocabulary comparisons include projected reference
models on identical positive rows, with original next-event labels preserved.
"""
import argparse,fcntl,gc,json,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from maomao.data.event_sequence import EventSequenceDataset,collate_event_sequences
from maomao.data.scale_ablation import ProjectedOutcomeDataset
from maomao.evaluation.softmax_calibration import fit_temperature,PROTOCOL,LEGACY_PROTOCOL
from maomao.evaluation.event_bias_calibration import fit_event_calibration,apply_calibration
from maomao.evaluation.event_metrics import event_metric_report_with_subsample_ci
from scripts.diagnostics.build_fullscale_external_rows import SOURCES
from scripts.diagnostics.evaluate_external_validation import check_contract
from scripts.diagnostics.scale_ablation_scope import model_for,NAMES
from scripts.diagnostics.uniform_result_scope import sha256
from scripts.diagnostics.maomao_prediction_only import omit_unused_heads

WORK=ROOT/'outputs/maomao_manuscript_figures_20260929'
OUT=WORK/'external_ablations';BASE=ROOT/'outputs/final_experiment_results_20260923'
ROWS=BASE/'classical_full_scale/external';TRAIN=ROOT/'data/perioperative_event_sequences_v5_richctx_static7'
SITES=('surgical_pooled','sicdb','mimic','mover','eicu')
MODULES=('no_block_causal','no_relative_time','no_family_head','no_clock_phase_summary',
         'no_measurement_intensity','no_masked_event_value','no_event_conditioned_time')
MODELS=(*MODULES,*NAMES,'reference_projected')
DEVICE=torch.device('cuda');VERSION='saved_maomao_ablation_full_external_90_10_v1'
def stamp():return datetime.now(timezone.utc).isoformat()
def read(p):return json.loads(p.read_text())
def write(p,a):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
 tmp.write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def checkpoint(name):
 if name in MODULES:return ROOT/'outputs/module_ablations_richctx_20260923'/name/'best_model.pt'
 if name=='reference_projected':return BASE/'full_maomao_reference/best_model.pt'
 return ROOT/'outputs/scale_ablations_richctx_20260928/runs'/name/'best_model.pt'
def guard():
 free=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
 if free<8*1024**3:raise RuntimeError('System MemAvailable below 8 GiB; stopped without discarding valid results')
def model_dataset(site,name):
 p=checkpoint(name);payload=torch.load(p,map_location='cpu',weights_only=False)
 d=EventSequenceDataset(ROOT/SOURCES[site],256,128,dynamic_windows=False)
 ids=list(range(210));cfg=payload['args']
 if cfg.get('outcome_projection_file'):
  d=ProjectedOutcomeDataset(d,cfg['outcome_projection_file']);ids=d.indices.tolist()
 model=model_for(d,payload,DEVICE)
 return d,model,ids,int(payload['epoch'])
def score_logits(site,name,split,folder,batch_size,state):
 source=ROWS/site;roww=np.load(source/f'{split}_window_indices.npy',mmap_mode='r')
 rowp=np.load(source/f'{split}_positions.npy',mmap_mode='r');truth=np.load(source/f'{split}_y.npy',mmap_mode='r')
 signature={'protocol':VERSION,'checkpoint_sha256':sha256(checkpoint(name)),
  'row_manifest_sha256':sha256(source/'manifest.json'),'rows':len(truth),'split':split,'batch_size':batch_size}
 p=folder/f'{split}_logits.npy';proofp=folder/f'{split}_prediction.json'
 if p.exists() and proofp.exists() and read(proofp).get('signature')==signature and read(proofp).get('complete'):
  return p,read(proofp)
 d,model,ids,epoch=model_dataset(site,name)
 shape=(len(truth),len(ids));windows=np.unique(roww)
 partial=folder/f'{split}_logits.partial.npy';progress=folder/f'{split}_prediction.partial.json'
 done=0
 if partial.exists() and progress.exists() and read(progress).get('signature')==signature:
  done=int(read(progress)['windows_completed']);z=np.load(partial,mmap_mode='r+')
  if z.shape!=shape:raise RuntimeError('Partial prediction shape changed')
 else:
  z=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,shape=shape)
 parameters=sum(p.numel() for p in model.parameters())
 # Verify exact event-logit equivalence before discarding unused output heads.
 probe=collate_event_sequences([d[int(w)] for w in windows[:min(2,len(windows))]])
 probe={k:v.to(DEVICE) for k,v in probe.items()}
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):full=model(probe).logits.clone()
 model=omit_unused_heads(model)
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):fast=model(probe).logits
 if not torch.equal(full,fast):raise RuntimeError('Prediction-only optimization changed frozen event logits')
 del probe,full,fast;torch.cuda.empty_cache()
 loader=DataLoader(Subset(d,windows[done:]),batch_size=batch_size,shuffle=False,num_workers=0,
  pin_memory=True,collate_fn=collate_event_sequences)
 began=time.monotonic()
 with torch.inference_mode():
  for b,batch in enumerate(loader):
   guard();at=done+b*batch_size;current=windows[at:at+len(batch['token_id'])]
   gathered=[];locals_=[];positions=[]
   for local,w in enumerate(current):
    lo=int(np.searchsorted(roww,w,'left'));hi=int(np.searchsorted(roww,w,'right'))
    pp=np.asarray(rowp[lo:hi],dtype=np.int64)
    actual=batch['target_set'][local,pp].numpy().astype(np.uint8)
    expected=np.asarray(truth[lo:hi][:,ids],dtype=np.uint8)
    if not np.array_equal(actual,expected):raise RuntimeError('Original external target coordinates changed')
    if not np.array_equal(batch['loss_mask'][local,pp].numpy(),expected.any(1)):
     raise RuntimeError('Projected valid-target mask differs from original next-event truth')
    gathered.extend(range(lo,hi));locals_.extend([local]*(hi-lo));positions.extend(pp.tolist())
   batch={k:v.to(DEVICE,non_blocking=True) for k,v in batch.items()}
   with torch.autocast('cuda',dtype=torch.bfloat16):logits=model(batch).logits.float()
   selected=logits[torch.tensor(locals_,device=DEVICE),torch.tensor(positions,device=DEVICE)].cpu().numpy()
   if not np.isfinite(selected).all():raise RuntimeError('Nonfinite frozen prediction')
   z[np.asarray(gathered)]=selected
   finished=at+len(current)
   if b%25==0 or finished==len(windows):
    z.flush();write(progress,{'signature':signature,'windows_completed':finished,'updated_utc':stamp()})
    state.update(status=f'predicting_{split}',site=site,model=name,windows_completed=finished,windows_total=len(windows),
      updated_utc=stamp(),elapsed_prediction_seconds=time.monotonic()-began);write(folder/'status.json',state)
    print(f'{site}/{name}/{split}: windows {finished}/{len(windows)}',flush=True)
   del logits,selected,batch
 z.flush();del z,model,d;gc.collect();torch.cuda.empty_cache();partial.replace(p)
 proof={'signature':signature,'complete':True,'all_original_rows_scored':True,'target_projection_verified':True,
  'prediction_only_bit_identical_verified':True,'indices':ids,'checkpoint_epoch':epoch,'parameter_count':parameters,
  'logits_sha256':sha256(p),'finished_utc':stamp()}
 write(proofp,proof);return p,proof
def project_arrays(zpath,ypath,folder,label,indices,source_indices):
 z=np.load(zpath,mmap_mode='r');y=np.load(ypath,mmap_mode='r')
 mask=np.zeros(len(y),bool)
 for lo in range(0,len(y),65536):mask[lo:lo+65536]=np.asarray(y[lo:lo+65536][:,indices]).any(1)
 positions=np.flatnonzero(mask);n=len(positions);cols=[source_indices.index(i) for i in indices]
 zp=folder/f'{label}_eligible_logits.npy';yp=folder/f'{label}_eligible_y.npy'
 zo=np.lib.format.open_memmap(zp,mode='w+',dtype=np.float32,shape=(n,len(indices)))
 yo=np.lib.format.open_memmap(yp,mode='w+',dtype=np.uint8,shape=(n,len(indices)))
 for lo in range(0,n,65536):
  rr=positions[lo:lo+65536];zo[lo:lo+65536]=z[rr][:,cols];yo[lo:lo+65536]=y[rr][:,indices]
 zo.flush();yo.flush();del zo,yo,z,y
 write(folder/f'{label}_eligible_rows.json',dict(original_rows=len(mask),eligible_rows=n,
  excluded_empty_projected_target=int((~mask).sum()),indices=indices,eligible_position_sha256=__import__('hashlib').sha256(positions.tobytes()).hexdigest()))
 return zp,yp,n
def report(site,name,folder,cal,test,proof,indices,label):
 destination=folder/label;destination.mkdir(parents=True,exist_ok=True)
 manifest=read(ROWS/site/'manifest.json');cp=checkpoint(name);modelhash=sha256(cp)
 expected={'model_sha256':modelhash,'row_manifest_sha256':sha256(ROWS/site/'manifest.json'),'output_indices':indices,'protocol':VERSION,'calibration_protocol':PROTOCOL}
 paths=[destination/f'metrics_{s}.json' for s in ('before','after')]
 if all(p.exists() and all(read(p).get(k)==v for k,v in expected.items()) for p in paths):return
 cz,cy,ncal=project_arrays(cal,ROWS/site/'calibration_y.npy',destination,'calibration',indices,proof['indices'])
 tz,ty,ntest=project_arrays(test,ROWS/site/'test_y.npy',destination,'test',indices,proof['indices'])
 if not ncal or not ntest:raise RuntimeError(f'No valid target rows for {site}/{label}')
 if PROTOCOL==LEGACY_PROTOCOL:fit=fit_temperature(cz,cy,DEVICE)
 else:
  from scripts.diagnostics.run_v5_bias_calibration_revision import row_groups
  truth=np.load(ROWS/site/'calibration_y.npy',mmap_mode='r');mask=np.empty(len(truth),bool)
  for lo in range(0,len(truth),65536):mask[lo:lo+65536]=truth[lo:lo+65536][:,indices].any(1)
  fit=fit_event_calibration(cz,cy,np.asarray(row_groups(site))[mask],DEVICE)
 write(destination/'calibration_fit.json',fit)
 del_caches=[cz,cy,tz,ty]
 outcomes=read(TRAIN/'event_sequence_meta.json')['outcome_vocabulary'];names=[outcomes[i] for i in indices]
 scores=torch.from_numpy(np.array(np.load(tz,mmap_mode='r'),copy=True));targets=torch.from_numpy(np.array(np.load(ty,mmap_mode='r'),copy=True))
 for s,t in [('before',1.),('after',fit['temperature'])]:
  if s=='after' and t==1. and not np.any(fit.get('bias',[0.])):
   result=dict(read(paths[0]))
  else:
   transformed=apply_calibration(scores,fit) if s=='after' else scores
   result=event_metric_report_with_subsample_ci(transformed,targets,names,bootstrap_repeats=200,
      bootstrap_seed=42,max_ci_rows=30000)
  result.update(expected,status='completed',site=site,model=label,calibration_state=s,temperature=t,
   calibration_protocol=PROTOCOL,calibration_rows=ncal,test_rows=ntest,
   calibration_rows_original=manifest['calibration_target_rows'],test_rows_original=manifest['test_target_rows'],
   split='frozen patient-disjoint 90:10 seed42',patients_calibration=manifest['split']['patients_validation'],
   patients_test=manifest['split']['patients_test'],patient_overlap=manifest['split']['patient_overlap'],
   train_rows_original=14128539,output_classes=len(indices),checkpoint_epoch=proof['checkpoint_epoch'],
   parameter_count=proof['parameter_count'],all_original_rows_scored=True,all_eligible_test_rows_evaluated=True,
   test_labels_used_for_fit_or_selection=False,checkpoint=str(cp.relative_to(ROOT)),
   before_after_on_identical_rows=True,finished_utc=stamp())
  if PROTOCOL!=LEGACY_PROTOCOL:
   result.update(bias=fit['bias'] if s=='after' else [0.]*len(indices),calibration_family=fit['family'] if s=='after' else 'raw',calibration_fit_sha256=sha256(destination/'calibration_fit.json'))
  write(destination/f'metrics_{s}.json',result)
  print(f'{site}/{label}/{s}: AP {result["micro_auprc"]:.6f}, ROC {result["micro_auroc"]:.6f}, n={ntest}',flush=True)
 del scores,targets;gc.collect()
 for p in del_caches:p.unlink()
def evaluate(site,name,batch_size):
 folder=OUT/site/name;folder.mkdir(parents=True,exist_ok=True);statepath=folder/'status.json'
 if statepath.exists() and read(statepath).get('status')=='completed':
  state=read(statepath)
  if state.get('checkpoint_sha256')!=sha256(checkpoint(name)):raise RuntimeError('Completed source checkpoint changed')
  for label in state['report_labels']:
   for s in ('before','after'):
    if read(folder/label/f'metrics_{s}.json').get('status')!='completed':raise RuntimeError('Completed queue evidence missing')
  return
 check_contract(ROOT/SOURCES[site],TRAIN)
 manifest=read(ROWS/site/'manifest.json');split=manifest['split']
 if split['patient_overlap'] or split['patients_test']!=round(.1*split['patients_total']):raise RuntimeError('Not frozen disjoint 90:10')
 state={'status':'starting','site':site,'model':name,'pid':os.getpid(),'started_utc':stamp()};write(statepath,state)
 cal,proof=score_logits(site,name,'calibration',folder,batch_size,state)
 test,testproof=score_logits(site,name,'test',folder,batch_size,state)
 if testproof['indices']!=proof['indices']:raise RuntimeError('Calibration and sealed target task mismatch')
 labels=[]
 tasks=[(name,proof['indices'])]
 if name=='reference_projected':tasks=[(f'reference_vocab_{k}',read(ROOT/f'outputs/scale_ablations_richctx_20260928/specifications/vocab_{k}.json')['indices']) for k in (50,100,150)]
 for label,indices in tasks:
  state.update(status='metrics_and_calibration',report_model=label,updated_utc=stamp());write(statepath,state)
  report(site,name,folder,cal,test,proof,indices,label);labels.append(label)
 state.update(status='completed',checkpoint_sha256=sha256(checkpoint(name)),report_labels=labels,finished_utc=stamp());write(statepath,state)
 cal.unlink();test.unlink()
 for p in folder.glob('*.partial.json'):p.unlink()
 print(f'Completed frozen external job {site}/{name}',flush=True)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--site',choices=SITES);parser.add_argument('--model',choices=MODELS);parser.add_argument('--batch-size',type=int,default=64);args=parser.parse_args()
 torch.set_num_threads(8);OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'queue.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  jobs=[(s,m) for s in (args.site,) if s for m in ((args.model,) if args.model else MODELS)] if args.site else [(s,m) for s in SITES for m in ((args.model,) if args.model else MODELS)]
  state={'status':'running','pid':os.getpid(),'jobs_total':len(jobs),'scope':VERSION,'sites':list(SITES),'started_utc':stamp(),'completed':[]}
  write(OUT/'queue_status.json',state)
  try:
   for site,name in jobs:
    state.update(current_site=site,current_model=name,updated_utc=stamp());write(OUT/'queue_status.json',state)
    evaluate(site,name,args.batch_size);state['completed'].append({'site':site,'model':name});write(OUT/'queue_status.json',state)
   state.update(status='completed',finished_utc=stamp());write(OUT/'queue_status.json',state)
  except BaseException as e:
   state.update(status='failed',error=f'{type(e).__name__}: {e}',finished_utc=stamp());write(OUT/'queue_status.json',state);raise
if __name__=='__main__':main()
