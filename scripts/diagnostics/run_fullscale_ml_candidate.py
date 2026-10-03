#!/usr/bin/env python3
"""Fit one classical multi-label candidate to every valid training target row."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from maomao.evaluation.event_metrics import event_metric_report_with_subsample_ci
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import GaussianNB
import torch

BASE=ROOT/'outputs/final_experiment_results_20260923/classical_full_scale'
OUTROOT=ROOT/'outputs/final_experiment_results_20260923/classical_ml_fullscale'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--model',choices=['decision_tree','random_forest','hist_gradient_boosting','lightgbm','catboost','xgboost','adaboost','gaussian_nb'],required=True);a=ap.parse_args()
 out=OUTROOT/a.model;out.mkdir(parents=True,exist_ok=True);t=time.monotonic()
 X=np.load(BASE/'train_X.npy',mmap_mode='r');Y=np.load(BASE/'train_y.npy',mmap_mode='r')
 z=np.load(ROOT/'outputs/baseline_comparison_richctx_fair/baseline_rows_internal.npz')
 xv=z['x_validation'];yv=z['y_validation'].astype(bool)
 if a.model=='decision_tree':
  model=DecisionTreeClassifier(max_depth=20,min_samples_leaf=50,random_state=42)
 elif a.model=='random_forest':
  # Fit a deliberately memory-bounded full-row forest. Earlier parallel
  # settings were OOM-killed before producing any metrics.
  model=RandomForestClassifier(n_estimators=10,max_depth=8,min_samples_leaf=100,max_features=0.01,n_jobs=1,random_state=42)
 elif a.model=='hist_gradient_boosting':
  model=MultiOutputClassifier(HistGradientBoostingClassifier(max_iter=10,max_leaf_nodes=15,max_bins=32,learning_rate=.1,early_stopping=False,random_state=42),n_jobs=1)
 elif a.model=='lightgbm':
  vendor=OUTROOT/'lightgbm_vendor'
  if vendor.exists(): sys.path.insert(0,str(vendor))
  import lightgbm as lgb
  model=MultiOutputClassifier(lgb.LGBMClassifier(n_estimators=10,num_leaves=7,max_depth=3,
      max_bin=15,min_child_samples=1000,learning_rate=.1,n_jobs=1,verbosity=-1,
      force_col_wise=True,random_state=42),n_jobs=1)
 elif a.model=='catboost':
  vendor=OUTROOT/'catboost_vendor'
  if vendor.exists(): sys.path.insert(0,str(vendor))
  from catboost import CatBoostClassifier
  model=MultiOutputClassifier(CatBoostClassifier(iterations=10,depth=3,learning_rate=.1,
      thread_count=1,verbose=False,allow_writing_files=False,random_seed=42),n_jobs=1)
 elif a.model=='xgboost':
  import xgboost as xgb
  model=MultiOutputClassifier(xgb.XGBClassifier(n_estimators=10,max_depth=3,learning_rate=.1,
      subsample=.25,colsample_bytree=.01,max_bin=16,tree_method='hist',n_jobs=1,
      verbosity=0,objective='binary:logistic',eval_metric='logloss',random_state=42),n_jobs=1)
 elif a.model=='adaboost':
  from sklearn.ensemble import AdaBoostClassifier
  model=MultiOutputClassifier(AdaBoostClassifier(
      estimator=DecisionTreeClassifier(max_depth=2,min_samples_leaf=100,random_state=42),
      n_estimators=10,learning_rate=.1,random_state=42),n_jobs=1)
 else:
  model=MultiOutputClassifier(GaussianNB(),n_jobs=4)
 print(json.dumps({'stage':'fit_start','model':a.model,'train_rows':len(Y),'feature_count':X.shape[1],'target_count':Y.shape[1]}),flush=True)
 model.fit(X,Y)
 raw=model.predict_proba(xv)
 if isinstance(raw,list):
  columns=[]
  for i,p in enumerate(raw):
   classes=model.classes_[i]
   columns.append(p[:,int(np.flatnonzero(classes==1)[0])] if 1 in classes else np.zeros(len(xv),np.float32))
  probabilities=np.column_stack(columns)
 else:
  classes=model.classes_
  probabilities=np.column_stack([p[:,int(np.flatnonzero(classes==1)[0])] if 1 in classes else np.zeros(len(xv),np.float32) for p in raw])
 scores=torch.from_numpy(np.log(np.clip(probabilities,1e-6,1-1e-6))-np.log1p(-np.clip(probabilities,1e-6,1-1e-6))).float()
 meta=json.loads((ROOT/'data/perioperative_event_sequences_v5_richctx_static7/event_sequence_meta.json').read_text())
 report=event_metric_report_with_subsample_ci(scores,torch.from_numpy(yv),meta['outcome_vocabulary'],bootstrap_repeats=200,bootstrap_seed=4220,max_ci_rows=20000)
 params=model.get_params(deep=False) if hasattr(model,'get_params') else {}
 report.update({'model':a.model,'status':'completed','train_rows_full':len(Y),'validation_rows':len(yv),'epochs_or_rounds':10 if a.model in ('hist_gradient_boosting','lightgbm','catboost','xgboost','adaboost') else None,'split':'patient-disjoint 90:10, seed 42','validation_sampling':'same saved 20k patient-held-out target rows as deep baselines','training_seconds':round(time.monotonic()-t,2),'fit_scope':'all valid training target rows','parameters':'depth-capped decision tree' if a.model=='decision_tree' else 'RandomForest: 10 trees, max_depth=8, min_samples_leaf=100, max_features=0.01, n_jobs=1' if a.model=='random_forest' else 'HistGradientBoosting: 10 iterations per outcome, n_jobs=1' if a.model=='hist_gradient_boosting' else 'LightGBM 4.7.0: 10 trees per outcome, num_leaves=7, max_depth=3, max_bin=15, min_child_samples=1000, n_jobs=1' if a.model=='lightgbm' else 'CatBoost 1.2.10: 10 iterations per outcome, depth=3, thread_count=1' if a.model=='catboost' else 'XGBoost 3.4.1: 10 trees per outcome, max_depth=3, max_bin=16, colsample_bytree=0.01, subsample=0.25, n_jobs=1' if a.model=='xgboost' else 'AdaBoost: 10 estimators per outcome, depth-2 decision-tree base estimator, n_jobs=1' if a.model=='adaboost' else 'one GaussianNB per outcome','estimator_parameters':{k:params.get(k) for k in ('n_estimators','max_depth','min_samples_leaf','max_features','n_jobs','max_iter','max_leaf_nodes','max_bins') if k in params}})
 (out/'metrics.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
 import joblib;joblib.dump(model,out/'model.joblib',compress=1)
 print(json.dumps({'stage':'complete','model':a.model,'train_rows_full':len(Y),'seconds':report['training_seconds'],'micro_auprc':report['micro_auprc'],'micro_auroc':report['micro_auroc']}),flush=True)
if __name__=='__main__':main()
