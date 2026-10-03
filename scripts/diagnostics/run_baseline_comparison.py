#!/usr/bin/env python3
"""Run non-MAOMAO baselines on the same sparse next-event task.

The baselines use the causal raw-event sequence contract used by MAOMAO. A row is
one valid next-event prediction position; targets remain multi-label because
several events can occur at the same timestamp.  Classical models receive a
fixed-length, left-padded flattening of causal raw sequence channels plus the
same static covariates. No hand-engineered phase or monitoring-intensity
features are added to these baselines.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from maomao.data.event_sequence import (EventSequenceDataset, collate_event_sequences)
from maomao.evaluation.event_metrics import event_metric_report_with_subsample_ci
from scripts.diagnostics.evaluate_external_validation import patient_split


def json_dump(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def patient_windows(dataset: EventSequenceDataset, validation_fraction: float,
                    seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    import pandas as pd
    admissions = pd.read_csv(dataset.path / "admissions.csv", usecols=["subject_id"])
    patients = admissions["subject_id"].astype(str).to_numpy()
    unique = np.unique(patients)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    n_val = max(1, int(round(len(unique) * validation_fraction)))
    val_patients = set(shuffled[:n_val].tolist())
    val_episode = np.asarray([p in val_patients for p in patients], dtype=bool)
    train = np.flatnonzero(~val_episode[dataset.window_admission])
    val = np.flatnonzero(val_episode[dataset.window_admission])
    return train, val, {
        "patients_total": int(len(unique)),
        "train_patients": int(len(unique) - n_val),
        "validation_patients": int(n_val),
        "patient_overlap": 0,
        "validation_fraction": float(validation_fraction),
        "seed": int(seed),
    }


RAW_SEQUENCE_CHANNELS = 6
BASELINE_SEQUENCE_LENGTH = 256


def row_features(sample: dict[str, torch.Tensor], position: int,
                 num_tokens: int) -> np.ndarray:
    """Flatten only the causal raw input available at ``position``.

    The left padding and channel order are fixed so linear/XGBoost baselines
    see the same token, kind, value, presence, absolute-time and gap inputs as
    the sequence model.  No phase state, family counts, rolling summary, or
    measurement-intensity feature is extracted specifically for these models.
    """
    start = max(0, int(position) - BASELINE_SEQUENCE_LENGTH + 1)
    token = sample["token_id"][start:position + 1].numpy().astype(np.float32)
    kind = sample["token_kind"][start:position + 1].numpy().astype(np.float32)
    value = sample["value"][start:position + 1].numpy().astype(np.float32)
    value = np.sign(value) * np.log1p(np.abs(value))
    value = np.clip(value, -12.0, 12.0)
    present = sample["has_value"][start:position + 1].numpy().astype(np.float32)
    absolute = sample["time_min"][start:position + 1].numpy().astype(np.float32)
    absolute = np.log1p(np.maximum(absolute, 0.0)) / math.log1p(43200.0)
    gap = sample["gap_min"][start:position + 1].numpy().astype(np.float32)
    gap = np.log1p(np.maximum(gap, 0.0)) / math.log1p(43200.0)
    sequence = np.stack((token / max(1, num_tokens - 1), kind / 16.0,
                         value / 12.0, present, absolute, gap), axis=1)
    padded = np.zeros((BASELINE_SEQUENCE_LENGTH, RAW_SEQUENCE_CHANNELS), dtype=np.float32)
    padded[-len(sequence):] = sequence
    static = sample["static"].numpy().astype(np.float32)
    current = padded[-1].copy()
    return np.concatenate((current, padded.reshape(-1), static)).astype(np.float32)


def extract_rows(dataset: EventSequenceDataset, indices: Iterable[int], limit: int,
                 seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample valid positions from a reproducibly shuffled window subset.

    Materialising every window would recompute trajectory labels for the whole
    189k-window corpus even though the baseline comparison only needs a fixed
    row budget.  Shuffling windows first preserves a broad patient/phase mix
    while allowing the extractor to stop as soon as the budget is reached.
    """
    rng = np.random.default_rng(seed)
    indices = np.asarray(list(indices), dtype=np.int64).copy()
    rng.shuffle(indices)
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for index in indices:
        sample = dataset[int(index)]
        valid = torch.nonzero(
            sample["loss_mask"] & (sample["target_set"].sum(-1) > 0),
            as_tuple=False).flatten().tolist()
        for position in valid:
            row = row_features(sample, int(position), dataset.num_tokens)
            target = sample["target_set"][int(position)].numpy().astype(np.uint8)
            features.append(row)
            targets.append(target)
            if limit > 0 and len(features) >= limit:
                return np.stack(features), np.stack(targets)
    if not features:
        raise RuntimeError("No valid next-event rows were extracted")
    return np.stack(features), np.stack(targets)


def logits_from_probabilities(probabilities: np.ndarray) -> torch.Tensor:
    clipped = np.clip(probabilities.astype(np.float32), 1e-6, 1 - 1e-6)
    return torch.from_numpy(np.log(clipped) - np.log1p(-clipped))


def univariate_baseline(x_train: np.ndarray, y_train: np.ndarray,
                        x_test: np.ndarray, num_tokens: int) -> np.ndarray:
    """P(event | current token), with a smoothed global fallback."""
    global_rate = (y_train.sum(0) + 1.0) / (len(y_train) + 2.0)
    ids = np.clip(np.rint(x_train[:, 0] * max(1, num_tokens - 1)).astype(int), 0, num_tokens - 1)
    test_ids = np.clip(np.rint(x_test[:, 0] * max(1, num_tokens - 1)).astype(int), 0, num_tokens - 1)
    rates = np.repeat(global_rate[None, :], num_tokens, axis=0)
    counts = np.ones(num_tokens, dtype=np.float64)
    for token_id, target in zip(ids, y_train):
        rates[token_id] += target
        counts[token_id] += 1.0
    rates /= counts[:, None] + 1.0
    return rates[test_ids]


def fit_and_score(name: str, probabilities: np.ndarray, targets: np.ndarray,
                  outcomes: list[str], bootstrap_repeats: int = 0,
                  bootstrap_seed: int = 42) -> dict:
    report = event_metric_report_with_subsample_ci(
        logits_from_probabilities(probabilities),
        torch.from_numpy(targets.astype(bool)), outcomes,
        bootstrap_repeats=bootstrap_repeats,
        bootstrap_seed=bootstrap_seed)
    report["model"] = name
    report["rows"] = int(len(targets))
    return report


def fit_temperature(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Fit one scalar temperature on the external calibration patients only."""
    logits = logits_from_probabilities(probabilities).float()
    labels = torch.from_numpy(targets.astype(np.float32))
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([log_temperature], lr=0.05)
    for _ in range(120):
        temperature = log_temperature.exp().clamp(0.15, 6.0)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return float(log_temperature.detach().exp().clamp(0.15, 6.0))


def calibrated_score(name: str, probabilities: np.ndarray, targets: np.ndarray,
                     outcomes: list[str], temperature: float,
                     bootstrap_repeats: int, bootstrap_seed: int) -> dict:
    logits = logits_from_probabilities(probabilities) / float(temperature)
    report = event_metric_report_with_subsample_ci(
        logits, torch.from_numpy(targets.astype(bool)), outcomes,
        bootstrap_repeats=bootstrap_repeats, bootstrap_seed=bootstrap_seed)
    report["model"] = name
    report["rows"] = int(len(targets))
    report["temperature"] = float(temperature)
    return report


def limit_windows(indices: np.ndarray, limit: int, seed: int) -> np.ndarray:
    """Deterministically sample a bounded window set without crossing splits."""
    indices = np.asarray(indices, dtype=np.int64)
    if limit <= 0 or len(indices) <= limit:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=limit, replace=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_dir", type=Path, default=Path("data/perioperative_event_sequences_v5_richctx_static7"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/baseline_comparison_v5"))
    parser.add_argument("--row_limit_train", type=int, default=50000)
    parser.add_argument("--row_limit_validation", type=int, default=20000)
    parser.add_argument("--row_limit_external", type=int, default=20000)
    parser.add_argument("--validation_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_classical", action="store_true",
                        help="Reuse existing univariate/logistic/XGBoost results.")
    parser.add_argument("--bootstrap_repeats", type=int, default=1000,
                        help="Bootstrap repetitions for AUROC 95%% CI; use 0 for smoke tests")
    parser.add_argument("--external_dirs", nargs="*", type=Path,
                        default=[Path("data/val_mimic_richctx_static7"), Path("data/val_mover_richctx_static7")])
    parser.add_argument("--external_split_dir", type=Path, default=Path("outputs/external_validation_v5"))
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = EventSequenceDataset(args.training_dir, 256, 128, dynamic_windows=False)
    train_windows, val_windows, split = patient_windows(dataset, args.validation_fraction, args.seed)
    if not args.skip_classical:
        x_train, y_train = extract_rows(dataset, train_windows, args.row_limit_train, args.seed)
        x_val, y_val = extract_rows(dataset, val_windows, args.row_limit_validation, args.seed + 1)
        np.savez_compressed(args.output_dir / "baseline_rows_internal.npz",
                            x_train=x_train, y_train=y_train, x_validation=x_val, y_validation=y_val)
    outcomes = dataset.meta["outcome_vocabulary"]
    prior_path = args.output_dir / "baseline_comparison.json"
    prior = json.loads(prior_path.read_text()) if args.skip_classical and prior_path.exists() else None
    if args.skip_classical and prior is None:
        raise FileNotFoundError("--skip_classical requires an existing baseline_comparison.json")
    if prior is not None:
        result = prior
        result["split"] = split
        result["outcomes"] = int(len(outcomes))
        result["models"] = {name: value for name, value in result.get("models", {}).items()
                             if name in {"univariate_last_token", "logistic_regression", "xgboost"}}
    else:
        result = {"protocol": "patient-disjoint sparse next-event comparison",
                  "feature_definition": "causal raw sequence flattening (token/kind/value/has_value/time/gap) + static",
                  "split": split, "train_rows": int(len(y_train)), "validation_rows": int(len(y_val)),
                  "outcomes": int(len(outcomes)), "models": {}}

    if not args.skip_classical:
        print("baseline stage=univariate", flush=True)
        uni = univariate_baseline(x_train, y_train, x_val, dataset.num_tokens)
        result["models"]["univariate_last_token"] = fit_and_score(
            "univariate_last_token", uni, y_val, outcomes, args.bootstrap_repeats, args.seed + 10)

        print("baseline stage=logistic_fit", flush=True)
        logistic = make_pipeline(StandardScaler(), OneVsRestClassifier(
            LogisticRegression(max_iter=100, solver="liblinear", C=1.0), n_jobs=4))
        logistic.fit(x_train, y_train)
        logistic_prob = logistic.predict_proba(x_val)
        result["models"]["logistic_regression"] = fit_and_score(
            "logistic_regression", logistic_prob, y_val, outcomes, args.bootstrap_repeats, args.seed + 11)

        try:
            from xgboost import XGBClassifier
            from sklearn.multioutput import MultiOutputClassifier
            print("baseline stage=xgboost_fit", flush=True)
            xgb = MultiOutputClassifier(XGBClassifier(
                n_estimators=30, max_depth=4, learning_rate=0.08, subsample=0.8,
                colsample_bytree=0.8, objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", n_jobs=8, random_state=args.seed), n_jobs=1)
            xgb.fit(x_train, y_train)
            print("baseline stage=xgboost_scored", flush=True)
            xgb_prob = np.column_stack([p[:, 1] if p.shape[1] > 1 else p[:, 0]
                                         for p in xgb.predict_proba(x_val)])
            result["models"]["xgboost"] = fit_and_score(
                "xgboost", xgb_prob, y_val, outcomes, args.bootstrap_repeats, args.seed + 12)
        except Exception as exc:
            result["models"]["xgboost"] = {"status": "failed", "error": repr(exc)}

    for external_dir in args.external_dirs:
        if not external_dir.exists():
            continue
        ext = EventSequenceDataset(external_dir, 256, 128, dynamic_windows=False)
        split_path = args.external_split_dir / f"{external_dir.name}_patient_split.npz"
        if split_path.exists():
            split_payload = np.load(split_path)
            calibration_indices = split_payload["validation_windows"]
            ext_indices = split_payload["test_windows"]
        else:
            # External evaluation is always patient-disjoint 90% calibration /
            # 10% sealed test.  Never fall back to evaluating on all patients.
            args.external_split_dir.mkdir(parents=True, exist_ok=True)
            calibration_indices, ext_indices, _ = patient_split(
                ext, external_dir, args.external_split_dir, 0.1, args.seed + 200)
        if not args.skip_classical:
            x_cal, y_cal = extract_rows(
                ext, calibration_indices, args.row_limit_external, args.seed + 101)
            x_ext, y_ext = extract_rows(ext, ext_indices, args.row_limit_external, args.seed + 100)
            print(f"baseline stage=external source={external_dir.name} "
                  f"calibration_rows={len(y_cal)} test_rows={len(y_ext)}", flush=True)
        ext_result = result.get("external_zero_shot", {}).get(external_dir.name,
                                                               {"models": {}})
        if not args.skip_classical:
            ext_result["rows"] = int(len(y_ext))
            ext_result["calibration_rows"] = int(len(y_cal))
        ext_result["protocol"] = "patient-disjoint 90% calibration / 10% sealed test"
        if not args.skip_classical:
            ext_probabilities = {
                "univariate_last_token": (
                    univariate_baseline(x_train, y_train, x_cal, dataset.num_tokens),
                    univariate_baseline(x_train, y_train, x_ext, dataset.num_tokens)),
                "logistic_regression": (logistic.predict_proba(x_cal), logistic.predict_proba(x_ext)),
            }
            if "xgb" in locals() and result["models"].get("xgboost", {}).get("status") != "failed":
                ext_probabilities["xgboost"] = (
                    np.column_stack([p[:, 1] if p.shape[1] > 1 else p[:, 0]
                                     for p in xgb.predict_proba(x_cal)]),
                    np.column_stack([p[:, 1] if p.shape[1] > 1 else p[:, 0]
                                     for p in xgb.predict_proba(x_ext)]))
            for offset, (name, (cal_prob, test_prob)) in enumerate(ext_probabilities.items()):
                temperature = fit_temperature(cal_prob, y_cal)
                ext_result["models"][name] = fit_and_score(
                    name, test_prob, y_ext, outcomes, args.bootstrap_repeats, args.seed + 20 + offset)
                ext_result.setdefault("calibrated_models", {})[name] = calibrated_score(
                    name, test_prob, y_ext, outcomes, temperature,
                    args.bootstrap_repeats, args.seed + 30 + offset)
        result.setdefault("external_zero_shot", {})[external_dir.name] = ext_result
        json_dump(result, args.output_dir / "baseline_comparison.json")
        print(f"baseline stage=external_saved source={external_dir.name}", flush=True)

    json_dump(result, args.output_dir / "baseline_comparison.json")
    print(json.dumps({"models": list(result["models"]), "output": str(args.output_dir / "baseline_comparison.json")}, indent=2))


if __name__ == "__main__":
    main()
