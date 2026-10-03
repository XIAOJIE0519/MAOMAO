#!/usr/bin/env python3
"""Structural and frozen-contract checks for external validation datasets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maomao.data.event_sequence import EventSequenceDataset  # noqa: E402


def validate(path: Path, training_dir: Path) -> dict:
    meta = json.loads((path / "event_sequence_meta.json").read_text())
    train = json.loads((training_dir / "event_sequence_meta.json").read_text())
    if meta.get("split") != "external_validation" or not meta.get("complete"):
        raise ValueError(f"{path}: not a complete external_validation dataset")
    for key in ("token_vocabulary", "outcome_vocabulary", "stored_outcome_vocabulary", "outcome_class_remap"):
        if meta.get(key) != train.get(key):
            raise ValueError(f"{path}: frozen training contract mismatch in {key}")

    ptr = np.load(path / "sequence_ptr.npy", mmap_mode="r")
    static = np.load(path / "static_baseline.npy", mmap_mode="r")
    total = int(ptr[-1])
    expected_static = int(train.get("num_static", 2))
    if (len(ptr) != meta["num_admissions"] + 1 or
            static.shape != (meta["num_admissions"], expected_static)):
        raise ValueError(f"{path}: pointer/static shape mismatch")
    sizes = {
        "token_id.bin": ("int32", 4), "time_min.bin": ("float32", 4),
        "value.bin": ("float32", 4), "has_value.bin": ("uint8", 1),
        "token_kind.bin": ("uint8", 1), "outcome_class.bin": ("int16", 2),
    }
    for name, (_, width) in sizes.items():
        actual = (path / name).stat().st_size
        if actual != total * width:
            raise ValueError(f"{path}/{name}: {actual} bytes, expected {total * width}")
    token = np.memmap(path / "token_id.bin", dtype="int32", mode="r", shape=(total,))
    times = np.memmap(path / "time_min.bin", dtype="float32", mode="r", shape=(total,))
    outcome = np.memmap(path / "outcome_class.bin", dtype="int16", mode="r", shape=(total,))
    if token.min() < 0 or token.max() >= len(train["token_vocabulary"]):
        raise ValueError(f"{path}: token IDs outside frozen vocabulary")
    if outcome.min() < -1 or outcome.max() >= len(train["stored_outcome_vocabulary"]):
        raise ValueError(f"{path}: outcome IDs outside frozen vocabulary")
    for index in np.linspace(0, len(ptr) - 2, min(2048, len(ptr) - 1), dtype=int):
        local = times[int(ptr[index]):int(ptr[index + 1])]
        if len(local) < 2 or local[0] != 0 or np.any(local[1:] < local[:-1]):
            raise ValueError(f"{path}: invalid chronology in episode {index}")
    dataset = EventSequenceDataset(path, block_size=256, window_stride=128)
    sample = dataset[0]
    return {
        "path": str(path), "episodes": meta["num_admissions"], "tokens": total,
        "windows": len(dataset), "sample_tokens": int(len(sample["token_id"])),
        "observed_outcomes": len(meta.get("audit_counts", {})),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--training_dir", type=Path, default=Path("data/perioperative_event_sequences_v5_full"))
    args = parser.parse_args()
    print(json.dumps([validate(path, args.training_dir) for path in args.paths], indent=2))


if __name__ == "__main__":
    main()
