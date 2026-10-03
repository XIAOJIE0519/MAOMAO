#!/usr/bin/env python3
"""Frozen MAOMAO representation probes for clinically transferable endpoints."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from maomao.data.event_sequence import EventSequenceDataset, collate_event_sequences  # noqa: E402
from maomao.evaluation.event_metrics import average_precision  # noqa: E402
from scripts.diagnostics.evaluate_external_validation import build_model, move_batch  # noqa: E402


TASK_EVENTS = {
    "AKI": ("aki_stage_1_signal", "aki_stage_2_signal", "aki_stage_3_signal"),
    "death": ("inhospital_death",),
    "transfusion": ("rbc_transfusion", "ffp_transfusion", "platelet_transfusion",
                    "cryo_transfusion"),
    "ICU_transfer": ("icu_transfer",),
}


@torch.no_grad()
def collect(model, dataset, indices, device, batch_size, max_examples):
    loader = DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False,
        num_workers=0, collate_fn=collate_event_sequences)
    names = dataset.meta["outcome_vocabulary"]
    task_indices = [[names.index(name) for name in events if name in names]
                    for events in TASK_EVENTS.values()]
    features, labels = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        horizon = len(dataset.trajectory_horizons_hours) - 1
        valid = batch["trajectory_mask"][..., horizon].bool()
        if not valid.any():
            continue
        features.append(output.hidden_state[valid].float().cpu())
        target = batch["trajectory_target"][..., horizon, :][valid].bool()
        labels.append(torch.stack([
            target[:, indices].any(1) if indices else torch.zeros(
                len(target), dtype=torch.bool, device=target.device)
            for indices in task_indices
        ], dim=1).float().cpu())
        if sum(len(value) for value in features) >= max_examples:
            break
    return torch.cat(features)[:max_examples], torch.cat(labels)[:max_examples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/event_maomao_v5_full/best_model.pt"))
    parser.add_argument("--data_dir", type=Path,
                        default=Path("data/perioperative_event_sequences_v5_full"))
    parser.add_argument("--split", type=Path,
                        default=Path("outputs/event_maomao_v5_full/patient_validation_split.npz"))
    parser.add_argument("--max_train_examples", type=int, default=100000)
    parser.add_argument("--max_validation_examples", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/event_maomao_v5_full/linear_probe.json"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["args"]
    dataset = EventSequenceDataset(
        args.data_dir, config.get("block_size", 256), config.get("window_stride", 128))
    split = np.load(args.split)
    model = build_model(dataset, checkpoint, device)
    model.requires_grad_(False)
    train_x, train_y = collect(
        model, dataset, split["train_window_indices"], device, args.batch_size,
        args.max_train_examples)
    val_x, val_y = collect(
        model, dataset, split["validation_window_indices"], device, args.batch_size,
        args.max_validation_examples)
    mean, std = train_x.mean(0), train_x.std(0).clamp_min(1e-5)
    train_x, val_x = (train_x - mean) / std, (val_x - mean) / std
    probe = nn.Linear(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-2, weight_decay=1e-3)
    train_x, train_y = train_x.to(device), train_y.to(device)
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(probe(train_x), train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        score = probe(val_x.to(device)).sigmoid().cpu()
    report = {
        "protocol": "frozen MAOMAO; linear probe; patient-disjoint internal validation",
        "train_examples": len(train_x), "validation_examples": len(val_x),
        "tasks": {
            name: {"positive_validation_examples": int(val_y[:, index].sum()),
                   "auprc": average_precision(score[:, index], val_y[:, index].bool())}
            for index, name in enumerate(TASK_EVENTS)
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
