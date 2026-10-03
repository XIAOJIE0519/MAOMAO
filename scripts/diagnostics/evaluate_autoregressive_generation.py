#!/usr/bin/env python3
"""Compare free-running MAOMAO event-time rollouts with held-out trajectories."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from maomao.data.event_sequence import EventSequenceDataset, collate_event_sequences  # noqa: E402
from maomao.evaluation.generation import generate_trajectory  # noqa: E402
from scripts.diagnostics.evaluate_external_validation import build_model, move_batch  # noqa: E402


def prefix_batch(batch: dict, end: int) -> dict:
    sequence_keys = {
        "token_id", "time_min", "gap_min", "value", "has_value", "token_kind",
        "phase_id", "observation_features", "attention_mask",
    }
    return {key: (value[:, :end] if key in sequence_keys else value)
            for key, value in batch.items()
            if key in sequence_keys or key in {"static", "history_family_counts"}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/event_maomao_v5_full/best_model.pt"))
    parser.add_argument("--data_dir", type=Path,
                        default=Path("data/perioperative_event_sequences_v5_full"))
    parser.add_argument("--split", type=Path,
                        default=Path("outputs/event_maomao_v5_full/patient_validation_split.npz"))
    parser.add_argument("--windows", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/event_maomao_v5_full/autoregressive_generation.json"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["args"]
    dataset = EventSequenceDataset(
        args.data_dir, config.get("block_size", 256), config.get("window_stride", 128))
    validation_indices = np.load(args.split)["validation_window_indices"][:args.windows]
    model = build_model(dataset, checkpoint, device)
    outcome_names = dataset.meta["outcome_vocabulary"]
    vocabulary = dataset.meta["token_vocabulary"]
    outcome_token_ids = [vocabulary[f"event:{name}"] for name in outcome_names]
    generated_counts = torch.zeros(dataset.num_outcomes)
    observed_counts = torch.zeros(dataset.num_outcomes)
    first_hits, time_errors = [], []
    evaluated = 0
    for index in validation_indices:
        batch = collate_event_sequences([dataset[int(index)]])
        valid_positions = batch["loss_mask"][0].nonzero(as_tuple=True)[0]
        if not len(valid_positions):
            continue
        position = int(valid_positions[len(valid_positions) // 2])
        true_next = batch["target_set"][0, position].bool()
        true_dt = float(batch["target_dt_hours"][0, position])
        observed_counts += batch["target_set"][0, position:].sum(0)
        context = move_batch(prefix_batch(batch, position + 1), device)
        rollout = generate_trajectory(
            model, context, outcome_token_ids, args.steps,
            max_context=config.get("block_size", 256))
        if not rollout:
            continue
        first_hits.append(bool(true_next[rollout[0]["event_index"]]))
        time_errors.append(abs(rollout[0]["delta_hours"] - true_dt))
        for item in rollout:
            generated_counts[item["event_index"]] += 1
        evaluated += 1
    p = generated_counts / generated_counts.sum().clamp_min(1)
    q = observed_counts / observed_counts.sum().clamp_min(1)
    midpoint = 0.5 * (p + q)
    js = 0.5 * ((p * (p.clamp_min(1e-12) / midpoint.clamp_min(1e-12)).log()).sum() +
                (q * (q.clamp_min(1e-12) / midpoint.clamp_min(1e-12)).log()).sum())
    report = {
        "protocol": "patient-validation free-running event->time->append rollout",
        "windows_evaluated": evaluated, "steps_per_rollout": args.steps,
        "first_event_hit_at_1": float(np.mean(first_hits)) if first_hits else None,
        "first_time_mae_hours": float(np.mean(time_errors)) if time_errors else None,
        "generated_vs_observed_event_distribution_js_divergence": float(js),
        "generated_event_counts": {name: int(generated_counts[index])
                                   for index, name in enumerate(outcome_names)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
