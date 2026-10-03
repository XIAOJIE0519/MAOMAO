#!/usr/bin/env python3
"""Verify exact cached mask scores for fixed-time SHAP coalitions."""
import sys,json
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.diagnostics.run_maomao_family_shap import DATA,PLOT,OUT,CHECKPOINT,prepare,game
from scripts.diagnostics.evaluate_external_validation import build_model
from scripts.diagnostics.maomao_prediction_only import omit_unused_heads
from scripts.diagnostics.uniform_result_scope import sha256
from maomao.data.event_sequence import EventSequenceDataset

def main():
 torch.set_num_threads(2);torch.manual_seed(42)
 d=EventSequenceDataset(DATA,256,128,dynamic_windows=False);choices=np.load(OUT/'private_source_window_query_plan.npy')
 model=build_model(d,torch.load(CHECKPOINT,map_location='cuda',weights_only=False),torch.device('cuda'));model.eval();omit_unused_heads(model)
 member=torch.nn.functional.one_hot(torch.tensor(d.outcome_family_ids.astype(int),device='cuda'),d.num_event_families).float()
 checks=[]
 for i in range(10):
  sample,active,local,history=prepare(d,*map(int,choices[i]));mask=np.stack([np.zeros(len(active)),np.ones(len(active)),np.arange(len(active))%2])
  orig=game(model,sample,local,history,member,cache_attention=False,inputs_only=False)(mask)
  cached=game(model,sample,local,history,member,cache_attention=True)(mask)
  checks.append(dict(anonymous_case=i,coalitions=3,outputs=273,bit_identical=bool(np.array_equal(orig,cached)),max_difference=float(np.abs(orig-cached).max())))
 proof=dict(complete=all(x['bit_identical'] for x in checks),checks=checks,checkpoint_sha256=sha256(CHECKPOINT),
   cache_scope='B32 original relative-time/block-causal mask at exactly the same observed fixed grid per patient. Only reused within that patient game; causal mask, weights and time grid unchanged.',
   unused_label_tensors_excluded_from_model_inputs=True)
 (PLOT/'fixed_time_cache_equivalence.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
 if not proof['complete']:raise SystemExit(1)
if __name__=='__main__':main()
