"""Audit full-data experiments before replacing the historical comparison."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.diagnostics.scale_ablation_scope import *


def audit():
    errors, pending = [], []
    training_evidence = {}
    manifest = read(OUT / "manifest.json")
    digest = sha256(OUT / "manifest.json")
    for name, evidence in manifest["source_evidence"].items():
        path = Path(name)
        if not path.is_file() or path.stat().st_size != evidence["size_bytes"] or sha256(path) != evidence["sha256"]:
            errors.append(f"Frozen source changed: {path}")
    for name, evidence in manifest.get("training_code_evidence", {}).items():
        if sha256(Path(name)) != evidence["sha256"]:
            errors.append(f"Training implementation changed during experiment: {name}")
    split = np.load(REFERENCE / "patient_validation_split.npz")
    if set(split["train_patients"]) & set(split["validation_patients"]) or len(split["validation_patients"]) != 9989:
        errors.append("Reference patient split is not the agreed disjoint 90:10")
    specs = manifest["output_specifications"]
    train_y = np.load(ROWS / "train_y.npy", mmap_mode="r")
    validation_y = np.load(ROWS / "validation_y.npy", mmap_mode="r")
    support = np.zeros(210, dtype=np.int64)
    for start in range(0, len(train_y), 100_000):
        support += train_y[start:start+100_000].sum(0, dtype=np.int64)
    ranked = sorted(range(210), key=lambda i: (-int(support[i]), i))
    for size in (50, 100, 150):
        spec = specs[f"vocab_{size}"]
        indices = sorted(ranked[:size])
        if (spec["indices"] != indices or spec["training_support"] != [int(support[i]) for i in indices] or
                read(OUT / f"specifications/vocab_{size}.json") != spec):
            errors.append(f"vocab_{size}: selection is not frozen from training rows alone")
        for label, arr in (("train", train_y), ("validation", validation_y)):
            count = sum(int(arr[start:start+100_000, indices].any(1).sum()) for start in range(0, len(arr), 100_000))
            if count != spec["eligible_rows"][label]:
                errors.append(f"vocab_{size}: incomplete {label} eligible target count")
    for name in ("reference", *NAMES, "reference_vocab_50", "reference_vocab_100", "reference_vocab_150"):
        path = OUT / "metrics" / f"{name}.json"
        scale_path = OUT / "metrics" / f"{name}_time_scales.json"
        if not path.exists() or not scale_path.exists():
            pending.append(name)
            continue
        metrics = read(path)
        size = int(name.rsplit("_", 1)[1]) if "vocab_" in name else 210
        spec = specs.get(f"vocab_{size}")
        rows = spec["eligible_rows"]["validation"] if spec else 1_563_972
        expected_indices = spec["indices"] if spec else list(range(210))
        is_reference = name.startswith("reference")
        train_eligible = 14_128_539 if is_reference or not spec else spec["eligible_rows"]["train"]
        checkpoint = REFERENCE / "best_model.pt" if is_reference else OUT / "runs" / name / "best_model.pt"
        if (metrics.get("status") != "completed" or metrics.get("event_targets") != rows or
                metrics.get("evaluation_rows") != rows or metrics.get("validation_rows_original") != 1_563_972 or
                metrics.get("train_rows_original") != 14_128_539 or metrics.get("output_indices") != expected_indices or
                metrics.get("train_event_rows_eligible") != train_eligible or
                metrics.get("train_output_classes_actual") != (210 if is_reference else size) or
                metrics.get("manifest_sha256") != digest or metrics.get("checkpoint_sha256") != sha256(checkpoint) or
                metrics.get("all_original_rows_scored") is not True or metrics.get("target_projection_verified") is not True):
            errors.append(f"{name}: row, output, or frozen model provenance mismatch")
        for key in METRICS:
            ci = metrics.get(key + "_95ci")
            if (not isinstance(metrics.get(key), (int, float)) or not math.isfinite(metrics[key]) or
                    not isinstance(ci, list) or len(ci) != 2 or not all(math.isfinite(v) for v in ci) or ci[0] > ci[1]):
                errors.append(f"{name}: {key} or 95% CI missing")
        if metrics.get("ci_rows") != 30_000 or metrics.get("ci_cohort_rows") != rows or metrics.get("ci_bootstrap_repeats") != 200:
            errors.append(f"{name}: approximate CI scope differs from current MAOMAO protocol")
        scales = read(scale_path)
        groups = scales.get("time_mae_by_scale", {})
        if (set(groups) != {"fine_0_to_2h", "long_2h_to_tail_start", "tail_from_tail_start"} or
                sum(g["event_target_rows"] for g in groups.values()) != rows or
                scales.get("checkpoint_sha256") != sha256(checkpoint) or scales.get("pooled_time_mae_reported") is not False or
                scales.get("output_indices") != expected_indices or scales.get("manifest_sha256") != digest or
                scales.get("target_time_sha256") != metrics.get("target_time_sha256") or scales.get("bootstrap_repeats") != 200):
            errors.append(f"{name}: time scales do not partition eligible full validation")
        for key, group in groups.items():
            if group["event_target_rows"] > 1 and (group.get("mae_hours") is None or not group.get("mae_hours_95ci")):
                errors.append(f"{name}/{key}: time MAE 95% CI missing")
        if not is_reference:
            config = read(checkpoint.parent / "run_config.json")
            if config != manifest["experiments"][name]:
                # A resumed run has the same training conditions; only its
                # bookkeeping resume path is allowed to differ.
                expected = dict(manifest["experiments"][name]); expected["resume"] = config.get("resume")
                if config != expected:
                    errors.append(f"{name}: fitting conditions changed beyond its declared experiment")
            history = [__import__("json").loads(line) for line in (checkpoint.parent / "validation_history.jsonl").read_text().splitlines()]
            updates_per_epoch = math.ceil(math.ceil(len(split["train_window_indices"]) / config["batch_size"]) / config["gradient_accumulation_steps"])
            if (len(history) != 7 or {row["epoch"] for row in history} != set(range(1, 8)) or
                    any(row["optimizer_step"] != row["epoch"] * updates_per_epoch or not math.isfinite(row["loss"]) for row in history)):
                errors.append(f"{name}: seven complete full-data epochs not proven")
            log = (checkpoint.parent / "train.log").read_text()
            if ("Training finished" not in log or
                    f"train_windows={len(split['train_window_indices'])} validation_windows={len(split['validation_window_indices'])}" not in log):
                errors.append(f"{name}: complete full-window training not proven")
            actual = np.load(checkpoint.parent / "patient_validation_split.npz")
            if any(not np.array_equal(split[key], actual[key]) for key in split.files):
                errors.append(f"{name}: patient/window split changed")
            training_evidence[name] = {
                "epochs_completed": len(history), "train_windows_per_epoch": len(split["train_window_indices"]),
                "validation_windows_per_epoch": len(split["validation_window_indices"]),
                "optimizer_updates_per_epoch": updates_per_epoch,
                "train_event_rows_per_epoch": train_eligible,
                "all_original_train_rows": 14_128_539, "seed": config["seed"],
                "batch_size": config["batch_size"], "gradient_accumulation_steps": config["gradient_accumulation_steps"],
                "training_log_sha256": sha256(checkpoint.parent / "train.log"),
                "validation_history_sha256": sha256(checkpoint.parent / "validation_history.jsonl"),
                "epoch_records": [{key: row[key] for key in ("epoch", "optimizer_step", "learning_rate", "loss", "event_loss", "family_loss")}
                                  for row in history]}
    for size in (50, 100, 150):
        a, b = OUT / "metrics" / f"vocab_{size}.json", OUT / "metrics" / f"reference_vocab_{size}.json"
        if a.exists() and b.exists():
            for key in ("event_targets", "output_indices", "manifest_sha256"):
                if read(a)[key] != read(b)[key]:
                    errors.append(f"vocab_{size}: reference is not scored on identical eligible rows/classes")
    ref = OUT / "metrics/reference.json"
    if ref.exists():
        expected_time_hash = read(ref).get("target_time_sha256")
        for name in NAMES:
            path = OUT / "metrics" / f"{name}.json"
            if path.exists() and (not expected_time_hash or read(path).get("target_time_sha256") != expected_time_hash):
                errors.append(f"{name}: original target-time coordinates differ from MAOMAO")
    return {"complete": not errors and not pending, "errors": errors, "pending": pending,
            "training_evidence_by_model": training_evidence}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = audit()
    write(OUT / "verification.json", result)
    print(__import__("json").dumps(result, ensure_ascii=False, indent=2))
    if result["errors"] or (args.require_complete and not result["complete"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
