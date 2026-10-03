"""Freeze the full-row experiment matrix and training-only event selection."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.diagnostics.scale_ablation_scope import *


def main():
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        print(f"Frozen manifest already exists: {manifest_path}")
        return
    train = np.load(ROWS / "train_y.npy", mmap_mode="r")
    valid = np.load(ROWS / "validation_y.npy", mmap_mode="r")
    if train.shape != (14_128_539, 210) or valid.shape != (1_563_972, 210):
        raise RuntimeError("Current full-row labels differ from the canonical comparison")
    names = read(DATA / "event_sequence_meta.json")["outcome_vocabulary"]
    counts = np.zeros(210, dtype=np.int64)
    for start in range(0, len(train), 100_000):
        counts += train[start:start + 100_000].sum(0, dtype=np.int64)
    ranked = sorted(range(210), key=lambda i: (-int(counts[i]), i))
    specifications = {}
    for size in (50, 100, 150):
        indices = sorted(ranked[:size])
        eligible = {}
        for split_name, arr in (("train", train), ("validation", valid)):
            eligible[split_name] = sum(int(arr[start:start+100_000, indices].any(1).sum())
                                       for start in range(0, len(arr), 100_000))
        specification = {"selection": "top event support on full 90% training target rows only; original class order",
                         "selection_source": str(ROWS / "train_y.npy"), "train_rows_source": len(train),
                         "indices": indices, "outcome_names": [names[i] for i in indices],
                         "training_support": [int(counts[i]) for i in indices],
                         "eligible_rows": eligible,
                         "target_policy": "project original next-event set; never shift to a later retained event"}
        write(OUT / f"specifications/vocab_{size}.json", specification)
        specifications[f"vocab_{size}"] = specification
    files = [path for path in DATA.iterdir() if path.is_file()]
    files += [REFERENCE / "patient_validation_split.npz", ROWS / "train_y.npy", ROWS / "validation_y.npy",
              ROWS / "validation_window_indices.npy", ROWS / "validation_positions.npy", REFERENCE / "best_model.pt"]
    evidence = {str(path): {"sha256": sha256(path), "size_bytes": path.stat().st_size,
                           "mtime_ns": path.stat().st_mtime_ns} for path in files}
    code = [ROOT / name for name in ("scripts/train.py", "maomao/models/event_maomao.py", "maomao/data/event_sequence.py",
                                    "maomao/data/scale_ablation.py", "maomao/data/sampling.py", "maomao/data/outcome_families.py")]
    write(manifest_path, {"protocol": "full windows, patient-disjoint 90:10 seed42, seven epochs, no row sampling",
                         "train_rows_full": len(train), "validation_rows_full": len(valid),
                         "reference_checkpoint": str(REFERENCE / "best_model.pt"),
                         "reference_checkpoint_sha256": sha256(REFERENCE / "best_model.pt"),
                         "data_dir": str(DATA), "source_evidence": evidence,
                         "training_code_evidence": {str(path): {"sha256": sha256(path), "mtime_ns": path.stat().st_mtime_ns} for path in code},
                         "experiments": {name: configuration(name) for name in NAMES},
                         "output_specifications": specifications,
                         "context_policy": "isolate non-overlapping 64/128-token segments within original 256-token windows at every layer and auxiliary pass; preserve every target coordinate, static inputs, original window-start history and per-token phase/intensity features"})
    print(f"Frozen {len(NAMES)} new experiments and existing MAOMAO reference: {manifest_path}")


if __name__ == "__main__":
    main()
