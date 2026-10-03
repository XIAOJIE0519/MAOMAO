#!/usr/bin/env python3
"""Materialize all calibration and sealed-test target rows for external sites."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from maomao.data.event_sequence import EventSequenceDataset
from scripts.diagnostics.build_fullscale_baseline_rows import extract_split
from scripts.diagnostics.evaluate_external_validation import check_contract, patient_split


SOURCES = {
    "mimic": Path("data/val_mimic_richctx_static7"),
    "mover": Path("data/val_mover_richctx_static7"),
    "eicu": Path("data/val_eicu_surgical_final_maomao_static7"),
    "sicdb": Path("data/val_sicdb_surgical_final_maomao_static7"),
    "ntuh": Path("outputs/surgery_part1_part2_maomao_validation_final_v7/maomao_ntuh110_ecg_derived_hr"),
    "asac": Path("outputs/surgery_part1_part2_maomao_validation_final_v7/maomao_auckland_asac_eds25_surgical"),
    "uq": Path("outputs/surgery_part1_part2_maomao_validation_final_v7/maomao_uq_vital_signs32_surgical"),
    "surgical_pooled": Path("outputs/surgery_three_source_uniform_20260928/maomao_combined_three_source"),
}


def count_valid_rows(dataset, indices: np.ndarray) -> int:
    count = 0
    for n, window in enumerate(indices, 1):
        sample = dataset[int(window)]
        count += int((sample["loss_mask"] & (sample["target_set"].sum(-1) > 0)).sum())
        if n % 10000 == 0:
            print(f"counted windows={n}/{len(indices)} rows={count}", flush=True)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", choices=tuple(SOURCES), default=list(SOURCES))
    args = parser.parse_args()
    out = ROOT / "outputs/final_experiment_results_20260923/classical_full_scale/external"
    out.mkdir(parents=True, exist_ok=True)
    all_status = {}
    training_dir = ROOT / "data/perioperative_event_sequences_v5_richctx_static7"
    for label in args.sites:
        source = ROOT / SOURCES[label]
        name = source.name
        check_contract(source, training_dir)
        site_out = out / label
        site_out.mkdir(exist_ok=True)
        manifest_path = site_out / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            needed = [site_out / f"{split}_{part}.npy"
                      for split in ("calibration", "test")
                      for part in ("X", "y", "window_indices", "positions")]
            if (previous.get("status") == "ready_for_full_scale_evaluation"
                    and previous.get("dataset") == name and all(path.exists() for path in needed)):
                all_status[label] = previous
                print(f"{label}: reusing complete full-row 90:10 materialization", flush=True)
                continue
        dataset = EventSequenceDataset(source, 256, 128, dynamic_windows=False)
        split_path = ROOT / "outputs/external_validation_richctx_maomao" / label / f"{name}_patient_split.npz"
        if split_path.exists():
            payload = np.load(split_path)
            calibration, test = payload["validation_windows"], payload["test_windows"]
            admissions = pd.read_csv(source / "admissions.csv", dtype={"subject_id": "string"})
            groups = admissions["subject_id"].fillna("missing:").astype(str).to_numpy()
            val_admissions = payload["validation_admissions"]
            test_admissions = payload["test_admissions"]
            split_summary = {"split_file": str(split_path), "seed": 42,
                             "patients_total": int(len(np.unique(groups))),
                             "patients_validation": int(len(np.unique(groups[val_admissions]))),
                             "patients_test": int(len(np.unique(groups[test_admissions]))),
                             "validation_windows": int(len(calibration)),
                             "test_windows": int(len(test)), "patient_overlap": 0}
        else:
            split_dir = out / "splits"
            split_dir.mkdir(exist_ok=True)
            calibration, test, split_summary = patient_split(dataset, source, split_dir, 0.1, 42)
        status = {"dataset": name, "protocol": "patient-disjoint 90% calibration / 10% sealed test",
                  "split": split_summary, "status": "counting"}
        manifest_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
        n_cal = count_valid_rows(dataset, calibration)
        n_test = count_valid_rows(dataset, test)
        status.update(calibration_target_rows=n_cal, test_target_rows=n_test, status="extracting")
        manifest_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
        extract_split(dataset, calibration, "calibration", site_out, n_cal)
        extract_split(dataset, test, "test", site_out, n_test)
        status["status"] = "ready_for_full_scale_evaluation"
        manifest_path.write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
        all_status[label] = status
        print(json.dumps(status, ensure_ascii=False), flush=True)
    all_status = {site: json.loads((out / site / "manifest.json").read_text())
                  for site in SOURCES if (out / site / "manifest.json").exists()}
    (out / "manifest.json").write_text(json.dumps(all_status, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
