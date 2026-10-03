#!/usr/bin/env python3
"""Prespecified first 25 cases: compare saved 500-evaluation SHAP with 1000."""
import sys,json
from pathlib import Path
import numpy as np
import torch,shap
from scipy.cluster.hierarchy import linkage
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.diagnostics.run_maomao_family_shap import DATA,BASE,PLOT,OUT,CHECKPOINT,prepare,game
from maomao.data.event_sequence import EventSequenceDataset
from scripts.diagnostics.evaluate_external_validation import build_model
from scripts.diagnostics.score_uniform_external_five import save_json
from scripts.diagnostics.uniform_result_scope import sha256

def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    d=EventSequenceDataset(DATA,256,128,dynamic_windows=False)
    choices=np.load(OUT/'private_source_window_query_plan.npy')
    ck=torch.load(CHECKPOINT,map_location='cuda',weights_only=False);model=build_model(d,ck,torch.device('cuda'));model.eval()
    member=torch.nn.functional.one_hot(torch.tensor(d.outcome_family_ids.astype(int),device='cuda'),d.num_event_families).float()
    rows=[];arrays={}
    for i,(w,q) in enumerate(choices[:25]):
        sample,active,local,history=prepare(d,int(w),int(q));predict=game(model,sample,local,history,member)
        with np.load(PLOT/f'patients/case_{i:05d}.npz') as a:v500=a['shap_values'];base=a['baseline_log_family_probability']
        coordinates=d.outcome_family_ids[active//3].astype(float)*10000+active
        tree=linkage(coordinates[:,None],method='complete')
        masker=shap.maskers.Partition(np.zeros((1,len(active))),clustering=tree)
        exp=shap.PartitionExplainer(predict,masker,seed=42+i)(np.ones((1,len(active))),max_evals=1000,batch_size=32,silent=True)
        v1000=exp.values[0];delta=v1000-v500
        rows.append(dict(anonymous_case=i,input_groups=len(active),mean_absolute_shap_difference=float(np.abs(delta).mean()),
            max_absolute_shap_difference=float(np.abs(delta).max()),baseline_max_difference=float(np.abs(exp.base_values[0]-base).max()),
            mean_abs_shap_500=float(np.abs(v500).mean()),mean_abs_shap_1000=float(np.abs(v1000).mean())))
        arrays[f'case_{i:05d}_values_1000']=v1000.astype(np.float32)
    np.savez_compressed(PLOT/'budget_sensitivity_values.npz',**arrays)
    save_json(PLOT/'budget_sensitivity.json',dict(status='completed',selection='First 25 anonymous cases in the prespecified seeded patient plan; not selected by effect or stability.',
        cases=25,baseline_budget=500,comparison_budget=1000,model_sha256=sha256(CHECKPOINT),results=rows,
        interpretation='Finite-budget hierarchical Partition SHAP approximations can vary; additivity alone does not establish convergence.'))
    print(json.dumps(rows,indent=2),flush=True)

if __name__=='__main__':main()
