#!/usr/bin/env python3
"""Score baseline and MAOMAO checkpoints on one patient-disjoint internal split."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from maomao.data.event_sequence import EventSequenceDataset, collate_event_sequences
from maomao.evaluation.event_metrics import event_metric_report_with_subsample_ci
from maomao.models.event_maomao import EventMAOMAO
from scripts.diagnostics.run_baseline_comparison import patient_windows
from scripts.diagnostics.run_gru_baseline import GRUBaseline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=("maomao", "gru"), required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--bootstrap_repeats", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_eval_rows", type=int, default=20_000)
    ap.add_argument("--full_validation", action="store_true",
                    help="Evaluate every valid target position in the 10% patient validation split")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EventSequenceDataset(args.data_dir, 256, 128, dynamic_windows=False)
    _, val_indices, split = patient_windows(dataset, 0.1, 42)
    loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size,
                        shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda",
                        collate_fn=collate_event_sequences)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if args.kind == "maomao":
        cfg = payload["args"]
        model = EventMAOMAO(
            dataset.num_tokens, dataset.num_outcomes, dataset.num_static,
            cfg.get("hidden_dim", 384), cfg.get("num_layers", 10),
            cfg.get("num_heads", 12), cfg.get("ffn_dim", 1536),
            cfg.get("dropout", 0.1), cfg.get("initial_event_interval_hours", 24.0),
            cfg.get("decoupled_time_head", False), cfg.get("enhanced_time_encoding", False),
            len(dataset.trajectory_horizons_hours), cfg.get("lognormal_time_head", False),
            outcome_family_ids=torch.as_tensor(dataset.meta["outcome_to_family"], dtype=torch.long),
            use_family_head=cfg.get("use_family_head", True),
            same_time_block_causal=cfg.get("same_time_block_causal", False),
            relative_time_attention=cfg.get("relative_time_attention", False),
            event_conditioned_time_head=cfg.get("event_conditioned_time_head", False),
            phase_memory=cfg.get("phase_memory", False),
            clock_phase_context=cfg.get("clock_phase_context", True),
            observation_intensity=cfg.get("observation_intensity", False),
            value_reconstruction=cfg.get("masked_value_loss_weight", 0.0) > 0,
            dual_timescale_time_head=cfg.get("dual_timescale_time_head", False),
            fine_time_bins=cfg.get("fine_time_bins", 24),
            long_time_bins=cfg.get("long_time_bins", 44),
        ).to(device)
        clock_ids = [int(i) for token, i in dataset.meta["token_vocabulary"].items()
                     if token == "<CLOCK>" or token.startswith("phase_summary:")]
        model.clock_phase_token_ids = torch.as_tensor(clock_ids, dtype=torch.long, device=device)
    elif args.kind == "gru":
        model = GRUBaseline(dataset.num_tokens, dataset.num_outcomes, dataset.num_static).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    if args.full_validation:
        full_dir = ROOT / "outputs/final_experiment_results_20260923/classical_full_scale"
        row_windows = np.load(full_dir / "validation_window_indices.npy", mmap_mode="r").astype(np.int64)
        row_positions = np.load(full_dir / "validation_positions.npy", mmap_mode="r").astype(np.int64)
        expected_targets = torch.from_numpy(np.asarray(
            np.load(full_dir / "validation_y.npy", mmap_mode="r"), dtype=np.float32))
    else:
        rows_path = ROOT / "outputs/baseline_comparison_richctx_fair/baseline_rows_internal.npz"
        shared = np.load(rows_path)
        row_windows = shared["validation_window_indices"].astype(np.int64)
        row_positions = shared["validation_positions"].astype(np.int64)
        expected_targets = torch.from_numpy(shared["y_validation"].astype(np.float32))
    # Score the neural models on the same (window, position) coordinates as the
    # classical validation arrays, whether the full or sampled mode is used.
    rows_by_window: dict[int, list[tuple[int, int]]] = {}
    for row, (window, position) in enumerate(zip(row_windows, row_positions)):
        rows_by_window.setdefault(int(window), []).append((row, int(position)))
    all_logits = torch.empty((len(row_windows), dataset.num_outcomes), dtype=torch.float32)
    all_targets = torch.empty_like(all_logits)
    amp = device.type == "cuda"
    with torch.inference_mode():
        for n, batch in enumerate(loader, 1):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
                output = model(batch)
            logits = output.logits if args.kind == "maomao" else output
            batch_window_ids = val_indices[(n - 1) * args.batch_size:n * args.batch_size]
            for local, window in enumerate(batch_window_ids):
                for row, position in rows_by_window.get(int(window), ()):
                    if not bool(batch["loss_mask"][local, position]) or not bool(batch["target_set"][local, position].sum() > 0):
                        raise RuntimeError(f"Shared row coordinate became invalid: window={window} position={position}")
                    all_logits[row] = logits[local, position].float().cpu()
                    all_targets[row] = batch["target_set"][local, position].cpu()
            if n % 100 == 0:
                print(f"eval batches={n}/{len(loader)}", flush=True)
    logits, targets = all_logits, all_targets
    if not torch.equal(targets.to(torch.uint8), expected_targets.to(torch.uint8)):
        raise RuntimeError("Neural evaluation targets differ from the classical shared-row targets")
    selected_rows = len(logits)
    report = event_metric_report_with_subsample_ci(
        logits, targets, dataset.meta["outcome_vocabulary"],
        bootstrap_repeats=args.bootstrap_repeats, bootstrap_seed=4200,
        max_ci_rows=30_000)
    report.update({"model": args.model_name or args.kind, "checkpoint": str(args.checkpoint.resolve()),
                   "checkpoint_epoch": payload.get("epoch"), "split": split,
                   "data_dir": str(args.data_dir), "validation_windows": len(val_indices),
                   "validation_rows_available": selected_rows,
                   "evaluation_rows": len(logits), "evaluation_rows_match_classical": True,
                   "evaluation_rows_full_patient_validation": args.full_validation,
                   "evaluation_row_selection": "full_validation" if args.full_validation else "shared_sample",
                   "evaluation_row_sampling_seed": None if args.full_validation else 43})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: report.get(k) for k in
                      ("model", "event_targets", "micro_auprc", "micro_auroc", "macro_auprc",
                       "macro_auroc", "mrr", "brier", "ece", "hit_at_1", "recall_at_5",
                       "recall_at_10", "ci_method", "ci_rows")}, indent=2))


if __name__ == "__main__":
    main()
