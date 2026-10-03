#!/usr/bin/env python3
"""Create patient-disjoint development (85%) and sealed test (15%) datasets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ARRAYS = {
    "token_id.bin": np.int32,
    "time_min.bin": np.float32,
    "value.bin": np.float32,
    "has_value.bin": np.uint8,
    "token_kind.bin": np.uint8,
    "outcome_class.bin": np.int16,
}


def write_subset(source: Path, output: Path, episode_indices: np.ndarray,
                 split_name: str) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    admissions = pd.read_csv(source / "admissions.csv")
    ptr = np.load(source / "sequence_ptr.npy", mmap_mode="r")
    lengths = np.asarray(ptr[episode_indices + 1] - ptr[episode_indices], dtype=np.int64)
    out_ptr = np.empty(len(episode_indices) + 1, dtype=np.int64)
    out_ptr[0] = 0
    np.cumsum(lengths, out=out_ptr[1:])
    np.save(output / "sequence_ptr.npy", out_ptr)
    total = int(out_ptr[-1])
    for filename, dtype in ARRAYS.items():
        source_array = np.memmap(source / filename, dtype=dtype, mode="r", shape=(int(ptr[-1]),))
        target = np.memmap(output / filename, dtype=dtype, mode="w+", shape=(total,))
        cursor = 0
        for episode in episode_indices:
            lo, hi = int(ptr[episode]), int(ptr[episode + 1])
            size = hi - lo
            target[cursor:cursor + size] = source_array[lo:hi]
            cursor += size
        target.flush()
        del target, source_array
    static = np.load(source / "static_baseline.npy", mmap_mode="r")
    np.save(output / "static_baseline.npy", np.asarray(static[episode_indices], dtype=np.float32))
    admissions.iloc[episode_indices].to_csv(output / "admissions.csv", index=False)

    meta = json.loads((source / "event_sequence_meta.json").read_text())
    stored_vocab = meta.get("stored_outcome_vocabulary", meta["outcome_vocabulary"])
    remap = np.asarray(meta.get("outcome_class_remap", list(range(len(stored_vocab)))), dtype=np.int16)
    raw_outcome = np.memmap(output / "outcome_class.bin", dtype=np.int16, mode="r", shape=(total,))
    valid = raw_outcome >= 0
    remapped = remap[raw_outcome[valid]]
    remapped = remapped[remapped >= 0]
    counts = np.bincount(remapped, minlength=len(meta["outcome_vocabulary"]))
    meta.update({
        "split": split_name,
        "num_admissions": int(len(episode_indices)),
        "num_operation_episodes": int(len(episode_indices)),
        "num_unique_admissions": int(admissions.iloc[episode_indices].hadm_id.nunique()),
        "num_tokens": total,
        "audit_counts": {name: int(counts[i]) for i, name in enumerate(meta["outcome_vocabulary"])},
        "patient_split_source": "prepare_classic_benchmark_split.py, patient-grouped seed=20260927",
    })
    (output / "event_sequence_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return {"episodes": int(len(episode_indices)), "sequence_tokens": total,
            "patients": int(admissions.iloc[episode_indices].subject_id.nunique())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path,
                        default=Path("data/perioperative_event_sequences_v5_richctx_static7"))
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/classic_score_comparison/independent_15pct_test"))
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    if not 0 < args.test_fraction < 0.4:
        raise ValueError("test-fraction must be in (0, 0.4)")
    if args.output.exists():
        raise FileExistsError(f"Preserving existing output: {args.output}")
    source_meta = json.loads((args.source / "event_sequence_meta.json").read_text())
    if source_meta.get("split") != "all_train" or not source_meta.get("complete"):
        raise ValueError("Source must be a complete all_train event sequence dataset")
    admissions = pd.read_csv(args.source / "admissions.csv")
    unique_patients = admissions.subject_id.astype(str).drop_duplicates().to_numpy()
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(unique_patients)
    n_test = max(1, int(round(len(shuffled) * args.test_fraction)))
    test_patients = set(shuffled[:n_test])
    patient_ids = admissions.subject_id.astype(str).to_numpy()
    test_mask = np.isin(patient_ids, list(test_patients))
    test_indices = np.flatnonzero(test_mask)
    development_indices = np.flatnonzero(~test_mask)
    args.output.mkdir(parents=True, exist_ok=False)
    dev = write_subset(args.source, args.output / "development", development_indices, "all_train")
    test = write_subset(args.source, args.output / "test", test_indices, "test")
    np.savez_compressed(
        args.output / "patient_partition.npz",
        development_patients=np.asarray(sorted(set(patient_ids[development_indices]))),
        test_patients=np.asarray(sorted(test_patients)),
        development_episode_indices=development_indices,
        test_episode_indices=test_indices,
    )
    report = {
        "seed": args.seed,
        "test_fraction_requested": args.test_fraction,
        "source_patients": int(len(unique_patients)),
        "development": dev,
        "test": test,
        "patient_overlap": len(set(patient_ids[development_indices]) & set(patient_ids[test_indices])),
        "development_internal_validation_fraction": args.test_fraction / (1.0 - args.test_fraction),
        "test_status": "sealed; do not use for checkpoint selection or calibration",
    }
    (args.output / "split_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
