#!/usr/bin/env python3
"""Compare actual coalition probabilities before/after unused-head pruning."""
import sys,json
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.diagnostics.run_maomao_family_shap import DATA,BASE,PLOT,OUT,CHECKPOINT,prepare,game
from scripts.diagnostics.evaluate_external_validation import build_model
from scripts.diagnostics.maomao_prediction_only import omit_unused_heads
from scripts.diagnostics.uniform_result_scope import sha256
from maomao.data.event_sequence import EventSequenceDataset

def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    d=EventSequenceDataset(DATA,256,128,dynamic_windows=False)
    choices=np.load(OUT/'private_source_window_query_plan.npy')
    model=build_model(d,torch.load(CHECKPOINT,map_location='cuda',weights_only=False),torch.device('cuda'));model.eval()
    member=torch.nn.functional.one_hot(torch.tensor(d.outcome_family_ids.astype(int),device='cuda'),d.num_event_families).float()
    # Reference all 273 outputs at fixed, unselected actual case coalitions.
    cases=list(range(10));baseline=[];prepared=[]
    for i in cases:
        sample,active,local,history=prepare(d,*map(int,choices[i]));prepared.append((sample,local,history))
        mask=np.stack([np.zeros(len(active)),np.ones(len(active)),np.arange(len(active))%2])
        baseline.append((mask,game(model,sample,local,history,member)(mask)))
    retained={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith(('encoder.','token_embedding.','outcome_head.','family_head.','relative_time_bias.','time_encoding.','kind_embedding.','value_projection.','static_projection.','phase_embedding.','observation_projection.','history_projection.'))}
    omit_unused_heads(model)
    changed=[k for k,v in retained.items() if not torch.equal(v,model.state_dict()[k].cpu())]
    checks=[]
    for i,(sample,local,history),(mask,expected) in zip(cases,prepared,baseline):
        actual=game(model,sample,local,history,member)(mask)
        checks.append(dict(anonymous_case=i,coalitions=3,outputs=273,bit_identical=bool(np.array_equal(actual,expected)),max_absolute_log_probability_difference=float(np.max(np.abs(actual-expected)))))
    proof=dict(complete=not changed and all(x['bit_identical'] for x in checks),retained_parameter_changes=changed,checks=checks,
        checkpoint_sha256=sha256(CHECKPOINT),retained_parameter_tensor_count=len(retained),
        source_forward_dependency='time/value/hazard/tail/trajectory output branches do not feed event logits; frozen encoder/outcome/family and input/context transformations retained.',
        purpose='Reduce unused time-head compute during event/family SHAP; no statistical protocol or game change.')
    (PLOT/'prediction_only_equivalence.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
    if not proof['complete']:raise SystemExit(1)
if __name__=='__main__':main()
