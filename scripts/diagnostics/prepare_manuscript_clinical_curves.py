#!/usr/bin/env python3
"""Exact endpoint curves on common complete cases with paired patient CIs."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.diagnostics.uniform_result_scope import sha256
SOURCE=ROOT/'outputs/maomao_plot_sources'
OUT=ROOT/'outputs/maomao_manuscript_figures_20260929/source_data'
def metrics_from_order(y,w,order,ends):
 a=y[order]*w[order];b=(1-y[order])*w[order]
 tp=np.cumsum(a)[ends];fp=np.cumsum(b)[ends]
 p=tp[-1];n=fp[-1]
 if p<=0 or n<=0:return np.nan,np.nan
 tpr=tp/p;fpr=fp/n
 auc=np.sum(np.diff(np.r_[0,fpr])*(tpr+np.r_[0,tpr[:-1]])/2)
 ap=np.sum(np.diff(np.r_[0,tpr])*np.divide(tp,tp+fp,out=np.ones_like(tp),where=tp+fp>0))
 return float(auc),float(ap)
def prepare():
 OUT.mkdir(parents=True,exist_ok=True)
 source=SOURCE/'clinical_score_predictions_no_identifiers.csv'
 frame=pd.read_csv(source);pairs=json.loads((SOURCE/'clinical_curves/pair_index.json').read_text())
 by={}
 for pair in pairs:by.setdefault(pair['endpoint'],[]).append(pair)
 rows=[];deltas=[];cohorts=[];curves={};dictionary={}
 for ei,(endpoint,items) in enumerate(by.items()):
  specs=[('MAOMAO',f'maomao_{endpoint}',1)]+[(x['comparator'],x['score_column'],x['comparator_risk_direction']) for x in items]
  columns=[f'label_{endpoint}','anonymous_patient_cluster']+[x[1] for x in specs]
  keep=frame[columns].notna().all(1)&np.isfinite(frame[columns].to_numpy(float)).all(1)
  selected=frame.loc[keep];y=selected[f'label_{endpoint}'].to_numpy(float)
  patients,clusters=np.unique(selected.anonymous_patient_cluster.to_numpy(),return_inverse=True)
  if len(np.unique(y))!=2:raise RuntimeError(f'{endpoint}: no two-class evidence')
  cohort=dict(endpoint=endpoint,source_episodes=len(frame),complete_episodes=len(selected),patients=len(patients),
      excluded_missing_any_comparator_or_target=int((~keep).sum()),events=int(y.sum()),prevalence=float(y.mean()))
  cohorts.append(cohort);dictionary[endpoint]=dict(cohort,series=[])
  vectors=[];orders=[];ends_all=[];points=[]
  for si,(name,col,direction) in enumerate(specs):
   score=selected[col].to_numpy(float)*direction
   order=np.argsort(-score,kind='stable');ends=np.r_[np.flatnonzero(np.diff(score[order])!=0),len(score)-1]
   tp=np.cumsum(y[order])[ends];fp=np.cumsum(1-y[order])[ends]
   auc,ap=metrics_from_order(y,np.ones(len(y)),order,ends)
   prefix=f'{endpoint}__{si}'
   curves[prefix+'__fpr']=np.r_[0,fp/fp[-1]];curves[prefix+'__tpr']=np.r_[0,tp/tp[-1]]
   curves[prefix+'__recall']=np.r_[0,tp/tp[-1]];curves[prefix+'__precision']=np.r_[1,tp/(tp+fp)]
   orders.append(order);ends_all.append(ends);points.append((auc,ap))
   dictionary[endpoint]['series'].append(dict(name=name,prefix=prefix,score_column=col,risk_direction=direction))
  # A single patient-cluster resample is shared by every curve for this endpoint.
  rng=np.random.default_rng(42+ei);boot=np.full((200,len(specs),2),np.nan)
  for repeat in range(200):
   count=np.bincount(rng.integers(0,len(patients),len(patients)),minlength=len(patients))
   weights=count[clusters].astype(float)
   for si in range(len(specs)):boot[repeat,si]=metrics_from_order(y,weights,orders[si],ends_all[si])
  points=np.array(points)
  for si,(name,col,direction) in enumerate(specs):
   for mi,metric in enumerate(('auroc','average_precision')):
    valid=np.isfinite(boot[:,si,mi]);interval=np.quantile(boot[valid,si,mi],[.025,.975])
    rows.append(dict(endpoint=endpoint,model=name,metric=metric,value=points[si,mi],
        ci_lower=interval[0],ci_upper=interval[1],n_episodes=len(y),n_patients=len(patients),events=int(y.sum()),valid_bootstrap_repeats=int(valid.sum())))
    if si:
     delta=boot[:,0,mi]-boot[:,si,mi];valid=np.isfinite(delta);ci=np.quantile(delta[valid],[.025,.975])
     deltas.append(dict(endpoint=endpoint,comparator=name,metric=metric,maomao_minus_comparator=points[0,mi]-points[si,mi],
       ci_lower=ci[0],ci_upper=ci[1],n_episodes=len(y),n_patients=len(patients),valid_bootstrap_repeats=int(valid.sum())))
  print(endpoint,cohort,'all curves common rows; paired patient bootstrap completed',flush=True)
 np.savez_compressed(OUT/'clinical_common_curves.npz',**curves)
 pd.DataFrame(rows).to_csv(OUT/'clinical_common_metrics.csv',index=False)
 pd.DataFrame(deltas).to_csv(OUT/'clinical_common_deltas.csv',index=False)
 pd.DataFrame(cohorts).to_csv(OUT/'clinical_common_cohorts.csv',index=False)
 (OUT/'clinical_curve_dictionary.json').write_text(json.dumps(dictionary,indent=2)+'\n')
 proof=dict(complete=True,endpoints=len(by),comparisons=len(pairs),source_sha256=sha256(source),
   cohort_policy='Endpoint-specific intersection of all finite requested comparators, MAOMAO and target; identical rows for all curves within endpoint',
   model_protocol='Separate frozen INSPIRE clinical-score 85:15 patient-disjoint study; not the primary internal 90:10 validation',
   ci_method='200 percentile patient-cluster bootstrap; paired MAOMAO-minus-comparator differences on common complete cases',
   calibration_or_training_changed=False,curve_type='Exact empirical ROC and step PR at every distinct score; AP is threshold-grouped average precision',
   files={p.name:sha256(p) for p in OUT.glob('clinical_*')})
 (OUT/'clinical_preparation.json').write_text(json.dumps(proof,indent=2)+'\n')
 return proof
if __name__=='__main__':print(json.dumps(prepare(),indent=2))
