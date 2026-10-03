#!/usr/bin/env python3
"""Patient-macro time error on the sealed external test splits.

The main evaluator reports a micro-average over prediction positions.  This
companion report deduplicates positions repeated by overlapping windows, then
averages absolute errors within each patient before giving every patient equal
weight in the final macro average.
"""
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

from maomao.data.event_sequence import EventSequenceDataset  # noqa: E402
from scripts.diagnostics.evaluate_external_validation import (  # noqa: E402
    build_model,
    make_loader,
    move_batch,
    point_prediction,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/event_maomao_multitask_500/best_model.pt"))
    parser.add_argument("--data_dirs", nargs="+", type=Path,
                        default=[Path("data/val_mimic"), Path("data/val_mover")])
    parser.add_argument("--evaluation_dir", type=Path,
                        default=Path("outputs/external_validation"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "patient_macro_mae_hours": float(values.mean()),
        "patient_median_mae_hours": float(np.median(values)),
        "patient_mae_p25_hours": float(np.quantile(values, 0.25)),
        "patient_mae_p75_hours": float(np.quantile(values, 0.75)),
        "patients_mae_le_1h_fraction": float((values <= 1.0).mean()),
        "patients_mae_le_3h_fraction": float((values <= 3.0).mean()),
    }


@torch.no_grad()
def evaluate_source(args: argparse.Namespace, data_dir: Path, checkpoint: dict,
                    device: torch.device, amp_dtype: torch.dtype) -> dict:
    dataset = EventSequenceDataset(
        data_dir, checkpoint["args"].get("block_size", 256),
        checkpoint["args"].get("window_stride", 128))
    report_path = args.evaluation_dir / f"{data_dir.name}_evaluation.json"
    report = json.loads(report_path.read_text())
    calibration = report["selected_calibration"]["time"]
    split = np.load(args.evaluation_dir / f"{data_dir.name}_patient_split.npz")
    test_windows = split["test_windows"]

    admissions = pd.read_csv(data_dir / "admissions.csv", dtype={"subject_id": "string"})
    group_column = "subject_id" if "subject_id" in admissions else "admission_id"
    patient_by_admission = admissions[group_column].fillna("missing:").astype(str).to_numpy()

    model = build_model(dataset, checkpoint, device)
    loader = make_loader(dataset, test_windows, args.batch_size, args.num_workers, device)
    max_wait = checkpoint["args"].get("max_wait_hours", 912.0)

    # key -> (largest local context position, raw error, calibrated error)
    # Choosing the largest local position retains the duplicate prediction with
    # the most preceding context when overlapping windows cover the same event.
    unique_errors: dict[tuple[int, float], tuple[int, float, float]] = {}
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=device.type == "cuda"):
            output = model(batch)
        event_valid = batch["loss_mask"].bool() & (batch["target_set"].sum(-1) > 0)
        time_valid = batch["time_mask"].bool()
        positions = time_valid.nonzero(as_tuple=False)
        if not len(positions):
            continue
        observed = event_valid[time_valid]
        if not observed.any():
            continue
        positions = positions[observed]
        mu = output.time_mu[time_valid].float()[observed]
        log_sigma = output.time_log_sigma[time_valid].float()[observed]
        truth = batch["target_dt_hours"][time_valid].float()[observed].clamp(
            1.0 / 60.0, max_wait)
        raw_prediction = point_prediction(mu, log_sigma, "raw_mean", 0.0, max_wait)
        calibrated_prediction = point_prediction(
            mu, log_sigma, calibration.get("point_family", "raw_mean"),
            calibration.get("point_log_scale", 0.0), max_wait)
        raw_error = (raw_prediction - truth).abs().cpu().numpy()
        calibrated_error = (calibrated_prediction - truth).abs().cpu().numpy()
        positions_cpu = positions.cpu().numpy()
        admission_indices = batch["admission_index"][positions[:, 0]].cpu().numpy()
        current_times = batch["time_min"][positions[:, 0], positions[:, 1]].cpu().numpy()
        for index, (row, local_position) in enumerate(positions_cpu):
            admission_index = int(admission_indices[index])
            key = (admission_index, float(current_times[index]))
            previous = unique_errors.get(key)
            candidate = (int(local_position), float(raw_error[index]),
                         float(calibrated_error[index]))
            if previous is None or candidate[0] > previous[0]:
                unique_errors[key] = candidate

    patient_errors: dict[str, dict[str, list[float]]] = {}
    for (admission_index, _), (_, raw_error, calibrated_error) in unique_errors.items():
        patient = patient_by_admission[admission_index]
        values = patient_errors.setdefault(patient, {"raw": [], "calibrated": []})
        values["raw"].append(raw_error)
        values["calibrated"].append(calibrated_error)

    rows = []
    for patient, errors in patient_errors.items():
        rows.append({
            "patient_id": patient,
            "observed_event_times": len(errors["raw"]),
            "raw_patient_mae_hours": float(np.mean(errors["raw"])),
            "calibrated_patient_mae_hours": float(np.mean(errors["calibrated"])),
        })
    frame = pd.DataFrame(rows).sort_values("patient_id")
    csv_path = args.evaluation_dir / f"{data_dir.name}_patient_time_errors.csv"
    frame.to_csv(csv_path, index=False)
    result = {
        "source": data_dir.name,
        "test_patients_with_observed_events": int(len(frame)),
        "deduplicated_observed_event_times": int(len(unique_errors)),
        "aggregation": "mean absolute error per patient, followed by equal-weight patient mean",
        "raw": describe(frame["raw_patient_mae_hours"].to_numpy()),
        "calibrated": describe(frame["calibrated_patient_mae_hours"].to_numpy()),
        "patient_csv": str(csv_path.resolve()),
    }
    del model, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for patient-level external evaluation")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    amp_dtype = (torch.bfloat16 if checkpoint["args"].get("precision", "bf16") == "bf16"
                 else torch.float16)
    results = []
    for data_dir in args.data_dirs:
        print(f"[{data_dir.name}] evaluating patient-level time error", flush=True)
        results.append(evaluate_source(args, data_dir, checkpoint, device, amp_dtype))
    output = args.evaluation_dir / "patient_time_error_summary.json"
    output.write_text(json.dumps({"sources": results}, ensure_ascii=False, indent=2))
    print(json.dumps({"sources": results}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
