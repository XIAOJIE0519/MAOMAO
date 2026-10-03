#!/usr/bin/env python3
"""Add 90/10 patient-level temperature calibration to a saved GRU baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from scripts.diagnostics.run_gru_baseline import GRUBaseline, run_loader, move
from scripts.diagnostics.evaluate_external_validation import patient_split
from maomao.data.event_sequence import EventSequenceDataset, collate_event_sequences
from maomao.evaluation.event_metrics import event_metric_report_with_subsample_ci


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    log_t = torch.zeros((), device=logits.device, requires_grad=True)
    opt = torch.optim.Adam([log_t], lr=0.05)
    for _ in range(200):
        t = log_t.exp().clamp(0.15, 6.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits / t, targets)
        opt.zero_grad(); loss.backward(); opt.step()
    return float(log_t.detach().exp().clamp(0.15, 6.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_dir', type=Path, required=True)
    ap.add_argument('--external_dirs', nargs='+', type=Path, required=True)
    ap.add_argument('--external_split_dir', type=Path, required=True)
    ap.add_argument('--bootstrap_repeats', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.output_dir / 'gru_best.pt', map_location=device, weights_only=False)
    result = json.loads((args.output_dir / 'gru_baseline.json').read_text())
    for data_dir in args.external_dirs:
        print(f"[gru:calibration] starting {data_dir.name}", flush=True)
        ds = EventSequenceDataset(data_dir, 256, 128, dynamic_windows=False)
        cal_idx, test_idx, split = patient_split(ds, data_dir, args.external_split_dir, 0.1, args.seed + 200)
        cache_path = args.output_dir / f'{data_dir.name}_gru_external_logits.pt'
        if cache_path.is_file():
            cached = torch.load(cache_path, map_location='cpu', weights_only=False)
            cal_logits, cal_targets = cached['cal_logits'], cached['cal_targets']
            test_logits, test_targets = cached['test_logits'], cached['test_targets']
            del cached
        else:
            cal_loader = DataLoader(Subset(ds, cal_idx), batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_event_sequences)
            test_loader = DataLoader(Subset(ds, test_idx), batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_event_sequences)
            model = GRUBaseline(ds.num_tokens, ds.num_outcomes, ds.num_static).to(device)
            model.load_state_dict(ckpt['model'])
            cal_logits, cal_targets = run_loader(model, cal_loader, device)
            test_logits, test_targets = run_loader(model, test_loader, device)
        print(f"[gru:calibration] fitting/scoring {data_dir.name} calibration_rows={len(cal_targets)} "
              f"test_rows={len(test_targets)}", flush=True)
        sample_n = min(200_000, len(cal_logits))
        gen = torch.Generator(device='cpu').manual_seed(args.seed + 3)
        sample_idx = torch.randperm(len(cal_logits), generator=gen)[:sample_n]
        temperature = fit_temperature(cal_logits[sample_idx].to(device), cal_targets[sample_idx].to(device)).__float__()
        raw = event_metric_report_with_subsample_ci(test_logits, test_targets, ds.meta['outcome_vocabulary'], bootstrap_repeats=args.bootstrap_repeats, bootstrap_seed=args.seed + 1)
        calibrated = event_metric_report_with_subsample_ci(test_logits / temperature, test_targets, ds.meta['outcome_vocabulary'], bootstrap_repeats=args.bootstrap_repeats, bootstrap_seed=args.seed + 2)
        calibrated['temperature'] = temperature
        result.setdefault('external', {})[data_dir.name] = {
            'protocol': 'patient-disjoint 90% calibration / 10% sealed test',
            'split': split,
            'calibration_rows': int(len(cal_targets)), 'test_rows': int(len(test_targets)),
            'raw': raw, 'calibrated': calibrated,
        }
        del cal_logits, cal_targets, test_logits, test_targets
    (args.output_dir / 'gru_baseline_calibrated.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps({'output': str(args.output_dir / 'gru_baseline_calibrated.json')}), flush=True)


if __name__ == '__main__':
    main()
