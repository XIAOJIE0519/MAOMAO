#!/usr/bin/env python3
"""Store the exact valid-window coordinates used by the classical comparison."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from maomao.data.event_sequence import EventSequenceDataset
from scripts.diagnostics.run_baseline_comparison import patient_windows


def coordinates(dataset, window_indices, limit: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.asarray(window_indices, dtype=np.int64).copy()
    rng.shuffle(indices)
    windows, positions, targets = [], [], []
    for window in indices:
        sample = dataset[int(window)]
        valid = torch.nonzero(sample["loss_mask"] & (sample["target_set"].sum(-1) > 0), as_tuple=False).flatten()
        for position in valid.tolist():
            windows.append(int(window))
            positions.append(int(position))
            targets.append(sample["target_set"][position].numpy().astype(np.uint8))
            if limit > 0 and len(windows) >= limit:
                return np.asarray(windows), np.asarray(positions), np.stack(targets)
    return np.asarray(windows), np.asarray(positions), np.stack(targets)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("data/perioperative_event_sequences_v5_richctx_static7"))
    ap.add_argument("--rows_npz", type=Path, default=Path("outputs/baseline_comparison_richctx_fair/baseline_rows_internal.npz"))
    ap.add_argument("--validation_fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    dataset = EventSequenceDataset(args.data_dir, 256, 128, dynamic_windows=False)
    train, validation, _ = patient_windows(dataset, args.validation_fraction, args.seed)
    stored = np.load(args.rows_npz)
    old = {key: stored[key] for key in stored.files}
    train_win, train_pos, train_y = coordinates(dataset, train, len(old["y_train"]), args.seed)
    val_win, val_pos, val_y = coordinates(dataset, validation, len(old["y_validation"]), args.seed + 1)
    if not np.array_equal(train_y, old["y_train"]):
        raise RuntimeError("Reconstructed classical training rows do not match saved targets")
    if not np.array_equal(val_y, old["y_validation"]):
        raise RuntimeError("Reconstructed classical validation rows do not match saved targets")
    old.update(train_window_indices=train_win, train_positions=train_pos,
               validation_window_indices=val_win, validation_positions=val_pos)
    tmp = args.rows_npz.with_suffix(args.rows_npz.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **old)
    tmp.replace(args.rows_npz)
    print(f"saved exact shared-row coordinates: train={len(train_win)} validation={len(val_win)}")


if __name__ == "__main__":
    main()
