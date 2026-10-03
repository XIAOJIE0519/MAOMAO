#!/usr/bin/env python3
"""Full structural audit for perioperative event sequence artifacts."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from maomao.data.event_sequence import EventSequenceDataset  # noqa: E402


DTYPES = {
    "token_id": "int32", "time_min": "float32", "value": "float32",
    "has_value": "uint8", "token_kind": "uint8", "outcome_class": "int16",
}


def validate(path: Path, sample_windows: int = 2048):
    meta = json.loads((path / "event_sequence_meta.json").read_text())
    errors = []
    if not meta.get("complete") or meta.get("version") not in (3, 4, 5):
        errors.append("dataset is not a complete compatible event-only artifact")
    if "stable_interval" in meta.get("outcome_vocabulary", []):
        errors.append("stable_interval must not be a v3 prediction target")
    if meta.get("version") == 4:
        outcome_count = len(meta.get("outcome_vocabulary", []))
        if not 50 <= outcome_count <= 100:
            errors.append(f"v4 must expose 50-100 outcome classes, found {outcome_count}")
        if "<MASK>" not in meta.get("token_vocabulary", {}):
            errors.append("v4 is missing the <MASK> token required by masked-event training")
        if meta.get("trajectory_horizons_hours") != [1.0, 6.0, 24.0]:
            errors.append("v4 trajectory horizons must be [1, 6, 24] hours")
    if meta.get("version") == 5:
        selection = meta.get("outcome_selection", {})
        if meta.get("split") != "external_validation":
            if selection.get("maximum_selected_outcomes") is not None:
                errors.append("v5 must not impose a maximum event-class cap")
            if selection.get("minimum_occurrences") != 100:
                errors.append("v5 occurrence threshold must be 100")
            if selection.get("minimum_independent_patients") != 50:
                errors.append("v5 independent-patient threshold must be 50")
        if "<CLOCK>" not in meta.get("token_vocabulary", {}):
            errors.append("v5 is missing the sparse clock/no-event token")
        required_phases = {
            "phase_summary:preop", "phase_summary:induction", "phase_summary:surgery",
            "phase_summary:emergence", "phase_summary:pacu", "phase_summary:icu",
        }
        if not required_phases.issubset(meta.get("token_vocabulary", {})):
            errors.append("v5 is missing one or more explicit phase-summary tokens")
        if not meta.get("outcome_to_family"):
            errors.append("v5 is missing hierarchical outcome-family mapping")
    # The all-train artifact has a strict provenance boundary: it may only be
    # built from train_data.  External-validation artifacts intentionally come
    # from MIMIC/MOVER and instead freeze the training token/outcome contract.
    is_external = meta.get("split") == "external_validation"
    if not is_external:
        source_root = meta.get("source_root", "")
        if meta.get("mimic_used") is not False or "mimic" in source_root.lower():
            errors.append("training source boundary does not exclude MIMIC")
        for item in meta.get("source_manifest", []):
            source = Path(source_root) / item["name"]
            if not source.is_file() or source.stat().st_size != item["size_bytes"]:
                errors.append(f"source manifest mismatch: {source}")
    elif not meta.get("source_name") or not meta.get("frozen_training_contract"):
        errors.append("external validation metadata lacks source/frozen-contract provenance")

    ptr = np.load(path / "sequence_ptr.npy", mmap_mode="r")
    count = int(meta["num_tokens"])
    if len(ptr) != int(meta["num_admissions"]) + 1 or int(ptr[0]) != 0 or int(ptr[-1]) != count:
        errors.append("sequence_ptr dimensions/endpoints are invalid")
    if np.any(np.diff(ptr) < 2):
        errors.append("one or more admissions lack BOS/discharge boundaries")
    episodes = pd.read_csv(path / "admissions.csv")
    if len(episodes) != int(meta.get("num_admissions", -1)):
        errors.append("admission table length mismatch")
    if meta.get("training_unit") == "operation_episode":
        if len(episodes) != int(meta.get("num_operation_episodes", -1)):
            errors.append("operation episode table length mismatch")
        operation_key = next(
            (name for name in ("op_id", "operation_id", "ep_idx") if name in episodes), None)
        if operation_key is None or episodes[operation_key].duplicated().any():
            errors.append("operation episodes lack a unique operation identifier")
        duration_key = next(
            (name for name in ("episode_length_min", "duration_min") if name in episodes), None)
        if duration_key is None or float(episodes[duration_key].max()) > 38 * 1440:
            errors.append("perioperative episode exceeds 7d pre + 1d operation + 30d post cap")

    arrays = {}
    for name, dtype in DTYPES.items():
        expected = count * np.dtype(dtype).itemsize
        actual = os.path.getsize(path / f"{name}.bin")
        if actual != expected:
            errors.append(f"{name}.bin has {actual} bytes, expected {expected}")
        arrays[name] = np.memmap(path / f"{name}.bin", dtype=dtype, mode="r", shape=(count,))

    vocab_size = len(meta["token_vocabulary"])
    stored_outcomes = meta.get("stored_outcome_vocabulary", meta["outcome_vocabulary"])
    outcome_counts = Counter()
    invalid_time = invalid_value = invalid_token = invalid_kind = invalid_outcome = 0
    nonmonotonic = 0
    chunk = 2_000_000
    for start in range(0, count, chunk):
        end = min(count, start + chunk)
        times = np.asarray(arrays["time_min"][start:end])
        values = np.asarray(arrays["value"][start:end])
        tokens = np.asarray(arrays["token_id"][start:end])
        kinds = np.asarray(arrays["token_kind"][start:end])
        outcomes = np.asarray(arrays["outcome_class"][start:end])
        invalid_time += int((~np.isfinite(times) | (times < 0)).sum())
        invalid_value += int((~np.isfinite(values)).sum())
        invalid_token += int(((tokens < 1) | (tokens >= vocab_size)).sum())
        invalid_kind += int((kinds > max(meta["token_kinds"].values())).sum())
        invalid_outcome += int(((outcomes < -1) | (outcomes >= len(stored_outcomes))).sum())
        valid = outcomes >= 0
        for index, amount in zip(*np.unique(outcomes[valid], return_counts=True)):
            outcome_counts[stored_outcomes[int(index)]] += int(amount)
    outcome_patient_sets = {name: set() for name in meta["outcome_vocabulary"]}
    stored_to_active = np.asarray(meta.get(
        "outcome_class_remap", list(range(len(stored_outcomes)))), dtype=np.int16)
    patient_column = "subject_id" if "subject_id" in episodes else None
    for ai in range(len(ptr) - 1):
        lo, hi = int(ptr[ai]), int(ptr[ai + 1])
        if int(arrays["token_id"][lo]) != int(meta["token_vocabulary"]["<BOS>"]):
            errors.append(f"episode {ai} does not start with BOS")
            break
        if np.any(np.diff(arrays["time_min"][lo:hi]) < 0):
            nonmonotonic += 1
        if patient_column:
            present = np.unique(np.asarray(arrays["outcome_class"][lo:hi]))
            present = present[present >= 0]
            mapped = stored_to_active[present]
            for active_index in np.unique(mapped[mapped >= 0]):
                outcome_patient_sets[meta["outcome_vocabulary"][int(active_index)]].add(
                    str(episodes.iloc[ai][patient_column]))
    if any((invalid_time, invalid_value, invalid_token, invalid_kind, invalid_outcome, nonmonotonic)):
        errors.append("one or more array-domain/chronology checks failed")

    dataset = EventSequenceDataset(path)
    rng = np.random.default_rng(42)
    selected = rng.choice(len(dataset), min(sample_windows, len(dataset)), replace=False)
    target_positions = positive_waits = time_positions = censored_positions = 0
    for index in selected:
        sample = dataset[int(index)]
        mask = sample["loss_mask"].numpy()
        time_mask = sample["time_mask"].numpy()
        target_positions += int(mask.sum())
        positive_waits += int((sample["target_dt_hours"].numpy()[mask] > 0).sum())
        time_positions += int(time_mask.sum())
        censored_positions += int((time_mask & ~mask).sum())
    if target_positions == 0 or positive_waits != target_positions:
        errors.append("sampled targets are absent or have non-positive waiting times")
    if dict(outcome_counts) != meta.get("audit_counts", {}):
        errors.append("metadata outcome counts do not match binary outcome classes")
    if meta.get("version") == 5 and not is_external:
        selection = meta["outcome_selection"]
        for name in meta["outcome_vocabulary"]:
            if outcome_counts[name] < selection["minimum_occurrences"]:
                errors.append(f"active outcome below occurrence threshold: {name}")
            if len(outcome_patient_sets[name]) < selection["minimum_independent_patients"]:
                errors.append(f"active outcome below patient threshold: {name}")

    lengths = np.diff(ptr)
    report = {
        "valid": not errors,
        "errors": errors,
        "admissions": int(meta["num_admissions"]),
        "training_unit": meta.get("training_unit", "admission"),
        "unique_admissions": int(meta.get("num_unique_admissions", meta["num_admissions"])),
        "tokens": count,
        "windows": len(dataset),
        "sequence_length": {
            "min": int(lengths.min()), "median": float(np.median(lengths)),
            "p95": float(np.quantile(lengths, 0.95)), "p99": float(np.quantile(lengths, 0.99)),
            "max": int(lengths.max()),
        },
        "array_checks": {
            "invalid_time": invalid_time, "invalid_value": invalid_value,
            "invalid_token": invalid_token, "invalid_kind": invalid_kind,
            "invalid_outcome": invalid_outcome, "nonmonotonic_admissions": nonmonotonic,
        },
        "outcome_counts_from_binary": dict(outcome_counts),
        "active_outcomes": meta["outcome_vocabulary"],
        "unavailable_outcomes": meta.get("unavailable_outcomes", {}),
        "sampled_target_positions": target_positions,
        "positive_wait_positions": positive_waits,
        "sampled_time_positions": time_positions,
        "sampled_right_censored_positions": censored_positions,
        "source": meta.get("source_root", meta.get("source_name")),
        "mimic_used": meta.get("mimic_used"),
        "max_episode_days": float(episodes[
            "episode_length_min" if "episode_length_min" in episodes else "duration_min"
        ].max() / 1440),
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/perioperative_event_sequences_v5_full")
    parser.add_argument("--sample_windows", type=int, default=2048)
    parser.add_argument("--report", default="outputs/event_data_audit_v5.json")
    args = parser.parse_args()
    report = validate(Path(args.data_dir), args.sample_windows)
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
