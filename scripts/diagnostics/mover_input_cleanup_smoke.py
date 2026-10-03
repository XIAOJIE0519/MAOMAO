#!/usr/bin/env python3
"""QA-only saved MAOMAO inference using processed MOVER inputs."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from maomao.data.event_sequence import EventSequenceDataset, collate_event_sequences
from scripts.diagnostics.evaluate_external_validation import build_model, move_batch, check_contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('before', 'after'), required=True)
    args = parser.parse_args()
    out = ROOT / 'outputs/mover_extract_cleanup_20261001'
    torch.set_num_threads(4)
    torch.manual_seed(42)
    train = ROOT / 'data/perioperative_event_sequences_v5_richctx_static7'
    source = ROOT / 'data/val_mover_richctx_static7'
    contract = check_contract(source, train)
    dataset = EventSequenceDataset(source, 256, 128, dynamic_windows=False)
    checkpoint = torch.load(ROOT / 'outputs/final_experiment_results_20260923/full_maomao_reference/best_model.pt', map_location='cpu', weights_only=False)
    device = torch.device('cuda')
    model = build_model(dataset, checkpoint, device)
    selected = [0, len(dataset)//2, len(dataset)-1]
    batch = move_batch(collate_event_sequences([dataset[i] for i in selected]), device)
    values = {k:v.detach().cpu().numpy() for k,v in batch.items() if torch.is_tensor(v)}
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        prediction = model(batch)
    for k in ('logits','family_logits','fine_hazard_logits','long_hazard_logits','time_mu','time_log_sigma','trajectory_logits'):
        values['prediction_'+k] = getattr(prediction,k).cpu().numpy()
    np.savez(out / f'mover_inference_{args.phase}.npz', **values)
    if args.phase == 'after':
        with np.load(out / 'mover_inference_before.npz') as original:
            assert set(original.files) == set(values)
            for key, value in values.items():
                a=original[key]
                assert a.shape == value.shape and a.dtype == value.dtype
                assert np.array_equal(np.ascontiguousarray(a).view(np.uint8), np.ascontiguousarray(value).view(np.uint8)), key
    result=dict(complete=True, phase=args.phase, source_dataset=str(source), contract=contract,
                windows=selected, arrays_checked=len(values), bitwise_equal_to_before=args.phase=='after',
                extracted_directory_exists=(ROOT/'data/MOVER/extracted').exists())
    (out / f'mover_inference_{args.phase}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
