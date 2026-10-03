#!/usr/bin/env python3
"""Materialize every valid next-event row with the shared causal baseline input.

This module is used by the external 90:10 row builder.  The feature layout is
the same as ``run_baseline_comparison.row_features``: current six channels,
256 left-padded causal six-channel tokens, then seven static values.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from scripts.diagnostics.run_baseline_comparison import (
    BASELINE_SEQUENCE_LENGTH,
    RAW_SEQUENCE_CHANNELS,
)


def _features_for_positions(sample: dict[str, torch.Tensor],
                            positions: np.ndarray, num_tokens: int) -> np.ndarray:
    token = sample["token_id"].numpy().astype(np.float32)
    kind = sample["token_kind"].numpy().astype(np.float32)
    value = sample["value"].numpy().astype(np.float32)
    value = np.clip(np.sign(value) * np.log1p(np.abs(value)), -12.0, 12.0)
    present = sample["has_value"].numpy().astype(np.float32)
    absolute = sample["time_min"].numpy().astype(np.float32)
    absolute = np.log1p(np.maximum(absolute, 0.0)) / math.log1p(43200.0)
    gap = sample["gap_min"].numpy().astype(np.float32)
    gap = np.log1p(np.maximum(gap, 0.0)) / math.log1p(43200.0)
    channels = np.stack((token / max(1, num_tokens - 1), kind / 16.0,
                         value / 12.0, present, absolute, gap), axis=1)
    padded = np.pad(channels, ((BASELINE_SEQUENCE_LENGTH - 1, 0), (0, 0)))
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, BASELINE_SEQUENCE_LENGTH, axis=0).transpose(0, 2, 1)
    causal = windows[positions].reshape(len(positions),
                                         BASELINE_SEQUENCE_LENGTH * RAW_SEQUENCE_CHANNELS)
    static = np.broadcast_to(sample["static"].numpy().astype(np.float32),
                             (len(positions), len(sample["static"])))
    return np.concatenate((channels[positions], causal, static), axis=1)


def extract_split(dataset, window_indices: np.ndarray, split: str,
                  output_dir: Path, expected_rows: int) -> None:
    """Write all valid rows in window order, matching the sequence model split."""
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_count = (BASELINE_SEQUENCE_LENGTH + 1) * RAW_SEQUENCE_CHANNELS + dataset.num_static
    specs = {
        "X": (np.float32, (expected_rows, feature_count)),
        "y": (np.uint8, (expected_rows, dataset.num_outcomes)),
        "window_indices": (np.int64, (expected_rows,)),
        "positions": (np.int16, (expected_rows,)),
    }
    paths = {name: output_dir / f"{split}_{name}.npy" for name in specs}
    temporary = {name: path.with_suffix(".partial.npy") for name, path in paths.items()}
    arrays = {
        name: np.lib.format.open_memmap(temporary[name], mode="w+", dtype=dtype, shape=shape)
        for name, (dtype, shape) in specs.items()
    }
    row = 0
    for n, window in enumerate(window_indices, 1):
        sample = dataset[int(window)]
        mask = sample["loss_mask"] & (sample["target_set"].sum(-1) > 0)
        positions = torch.nonzero(mask, as_tuple=False).flatten().numpy().astype(np.int64)
        count = len(positions)
        if count:
            end = row + count
            if end > expected_rows:
                raise RuntimeError(f"{split}: row count exceeded allocation at window {n}")
            arrays["X"][row:end] = _features_for_positions(sample, positions, dataset.num_tokens)
            arrays["y"][row:end] = sample["target_set"][positions].numpy().astype(np.uint8)
            arrays["window_indices"][row:end] = int(window)
            arrays["positions"][row:end] = positions.astype(np.int16)
            row = end
        if n % 10_000 == 0 or n == len(window_indices):
            print(f"extract {split}: windows={n}/{len(window_indices)} rows={row}/{expected_rows}",
                  flush=True)
    if row != expected_rows:
        raise RuntimeError(f"{split}: extracted {row} rows, expected {expected_rows}")
    for array in arrays.values():
        array.flush()
    del arrays
    for name, path in paths.items():
        temporary[name].replace(path)
