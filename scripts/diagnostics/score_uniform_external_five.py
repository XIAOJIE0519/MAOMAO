#!/usr/bin/env python3
"""Score five frozen full-fit models on all external 90:10 target rows."""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from maomao.data.event_sequence import EventSequenceDataset, collate_event_sequences
from maomao.evaluation.event_metrics import event_metric_report, event_metric_report_with_subsample_ci
from maomao.evaluation.softmax_calibration import PROTOCOL
from scripts.diagnostics.build_fullscale_external_rows import SOURCES
from scripts.diagnostics.evaluate_external_validation import build_model, check_contract
from scripts.diagnostics.run_requested_50k_models import SequenceNet
from scripts.diagnostics.uniform_result_scope import current_xgboost_model, xgboost_revision, sha256

BASE = ROOT / "outputs/final_experiment_results_20260923/classical_full_scale/external"
INTERNAL = ROOT / "outputs/final_experiment_results_20260923"
RESULTS = ROOT / "outputs/external_validation_final_maomao_uniform"
TRAINING = ROOT / "data/perioperative_event_sequences_v5_richctx_static7"
MODELS = ("univariate", "logistic_regression", "xgboost", "ann", "maomao")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def arrays(site_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.load(site_dir / f"{split}_X.npy", mmap_mode="r")
    y = np.load(site_dir / f"{split}_y.npy", mmap_mode="r")
    if x.shape != (len(y), 1549) or y.shape[1] != 210:
        raise RuntimeError(f"Bad full-row arrays: {site_dir.name}/{split} {x.shape} {y.shape}")
    return x, y


def predict_flat(name: str, site_dir: Path, split: str, output: np.memmap) -> None:
    x, _ = arrays(site_dir, split)
    model = None
    scaler = None
    rates = None
    booster = None
    if name == "univariate":
        with np.load(INTERNAL / "baseline_metrics/common_full_validation/univariate_last_token_model.npz") as saved:
            if int(saved["training_rows"]) != 14_128_539:
                raise RuntimeError("Univariate fit did not use all internal training rows")
            rates = np.asarray(saved["token_rates"], dtype=np.float32)
    elif name == "logistic_regression":
        payload = torch.load(INTERNAL / "baseline_metrics/logistic_fullscale.pt",
                             map_location="cpu", weights_only=False)
        if int(payload["training_rows"]) != 14_128_539:
            raise RuntimeError("Logistic fit did not use all internal training rows")
        model = torch.nn.Linear(1549, 210).to(DEVICE)
        model.weight.data.copy_(payload["weight"].to(DEVICE))
        model.bias.data.copy_(payload["bias"].to(DEVICE))
        model.eval()
        with np.load(INTERNAL / "baseline_metrics/logistic_scaler.npz") as saved:
            scaler = (torch.as_tensor(saved["mean"], device=DEVICE, dtype=torch.float32),
                      torch.as_tensor(saved["scale"], device=DEVICE, dtype=torch.float32).clamp_min(1e-8))
    elif name == "xgboost":
        import xgboost as xgb
        booster = xgb.Booster(params={"nthread": 4})
        booster.load_model(str(current_xgboost_model()))
    elif name == "ann":
        payload = torch.load(INTERNAL / "baseline_metrics/common_full_validation/ann_fullscale_refined.pt",
                             map_location="cpu", weights_only=False)
        if int(payload["train_rows_full"]) != 14_128_539:
            raise RuntimeError("ANN fit did not use all internal training rows")
        model = SequenceNet("ann", 0, 210).to(DEVICE)
        model.load_state_dict(payload["model"])
        model.eval()
    else:
        raise ValueError(name)

    step = 8192 if name in ("logistic_regression", "ann") else 16_384
    for start in range(0, len(x), step):
        end = min(start + step, len(x))
        xb = np.asarray(x[start:end], dtype=np.float32)
        if name == "univariate":
            ids = np.clip(np.rint(xb[:, 0] * (len(rates) - 1)).astype(np.int64), 0, len(rates) - 1)
            p = np.clip(rates[ids], 1e-6, 1 - 1e-6)
            score = np.log(p) - np.log1p(-p)
        elif name == "xgboost":
            import xgboost as xgb
            score = booster.predict(xgb.DMatrix(xb, nthread=4), output_margin=True)
        else:
            with torch.inference_mode():
                tensor = torch.as_tensor(np.array(xb, copy=True), device=DEVICE)
                if name == "logistic_regression":
                    tensor.sub_(scaler[0]).div_(scaler[1])
                    score = model(tensor).float().cpu().numpy()
                else:
                    score = model(tensor, None, None)[0].float().cpu().numpy()
        output[start:end] = np.asarray(score, dtype=output.dtype)
        if end % (step * 100) == 0 or end == len(x):
            output.flush()
            print(f"{site_dir.name} {name} {split}: {end}/{len(x)} rows", flush=True)
    del model, booster, x
    gc.collect()


def predict_maomao(site: str, site_dir: Path, split: str, output: np.memmap,
                 batch_size: int) -> None:
    dataset = EventSequenceDataset(ROOT / SOURCES[site], 256, 128, dynamic_windows=False)
    checkpoint = torch.load(INTERNAL / "full_maomao_reference/best_model.pt",
                            map_location=DEVICE, weights_only=False)
    model = build_model(dataset, checkpoint, DEVICE)
    row_windows = np.load(site_dir / f"{split}_window_indices.npy", mmap_mode="r")
    row_positions = np.load(site_dir / f"{split}_positions.npy", mmap_mode="r")
    if len(row_windows) != len(output) or np.any(row_windows[1:] < row_windows[:-1]):
        raise RuntimeError(f"{site}/{split}: row windows are missing or unsorted")
    windows = np.unique(row_windows)
    loader = DataLoader(Subset(dataset, windows), batch_size=batch_size,
                        shuffle=False, num_workers=0, pin_memory=DEVICE.type == "cuda",
                        collate_fn=collate_event_sequences)
    with torch.inference_mode():
        for batch_number, batch in enumerate(loader, 1):
            batch = {key: value.to(DEVICE, non_blocking=True) if torch.is_tensor(value) else value
                     for key, value in batch.items()}
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                                enabled=DEVICE.type == "cuda"):
                logits = model(batch).logits.float()
            current_windows = windows[(batch_number - 1) * batch_size:batch_number * batch_size]
            for local, window in enumerate(current_windows):
                lo = int(np.searchsorted(row_windows, window, side="left"))
                hi = int(np.searchsorted(row_windows, window, side="right"))
                positions = np.asarray(row_positions[lo:hi], dtype=np.int64)
                if not bool(batch["loss_mask"][local, positions].all()):
                    raise RuntimeError(f"{site}/{split}: invalid target positions at window {window}")
                output[lo:hi] = logits[local, positions].cpu().numpy().astype(output.dtype)
            if batch_number % 50 == 0 or batch_number == len(loader):
                output.flush()
                print(f"{site} maomao {split}: {batch_number}/{len(loader)} batches", flush=True)
    del model, checkpoint, loader, dataset
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def make_logits(site: str, name: str, split: str, site_dir: Path,
                result_dir: Path, batch_size: int) -> Path:
    suffix = f"_{xgboost_revision()[:12]}" if name == "xgboost" else ""
    path = result_dir / f"{split}{suffix}_logits.npy"
    _, y = arrays(site_dir, split)
    if path.exists():
        saved = np.load(path, mmap_mode="r")
        if saved.shape == (len(y), 210):
            return path
        raise RuntimeError(f"Bad saved logits shape: {path}")
    temporary = result_dir / f"{split}{suffix}_logits.partial.npy"
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32,
        shape=(len(y), 210))
    if name == "maomao":
        predict_maomao(site, site_dir, split, output, batch_size)
    else:
        predict_flat(name, site_dir, split, output)
    output.flush()
    del output
    temporary.replace(path)
    return path


def fit_temperature(logits_path: Path, labels_path: Path) -> tuple[float, int]:
    """Fit softmax temperature on the complete 90% calibration split."""
    from maomao.evaluation.softmax_calibration import fit_temperature as fit
    result = fit(logits_path, labels_path, DEVICE)
    save_json(logits_path.parent / "calibration_fit.json", result)
    return result["temperature"], result["calibration_rows"]


def save_maomao_uncalibrated(site, site_dir, result_dir, manifest, outcomes, batch_size, calibrated):
    """Reuse the completed temperature fit, score the identical sealed holdout raw."""
    raw_path = result_dir / "metrics_uncalibrated.json"
    model_hash = sha256(INTERNAL / "full_maomao_reference/best_model.pt")
    reference_hash = sha256(result_dir / "metrics.json")
    if raw_path.exists():
        previous = json.loads(raw_path.read_text())
        if (previous.get("model_sha256") == model_hash and
                previous.get("calibrated_reference_sha256") == reference_hash and
                previous.get("test_rows") == manifest["test_target_rows"]):
            return previous
    raw_status = result_dir / "uncalibrated_status.json"
    save_json(raw_status, {"status":"scoring_identical_test_rows", "started_utc":stamp()})
    path = make_logits(site, "maomao", "test", site_dir, result_dir, batch_size)
    scores = torch.from_numpy(np.array(np.load(path, mmap_mode="r"), dtype=np.float32, copy=True))
    targets = torch.from_numpy(np.array(np.load(site_dir / "test_y.npy", mmap_mode="r"), dtype=np.uint8, copy=True))
    save_json(raw_status, {"status":"checking_calibrated_reference", "updated_utc":stamp()})
    from maomao.evaluation.event_bias_calibration import apply_calibration
    check = event_metric_report(apply_calibration(scores,calibrated), targets, outcomes)
    keys = ("micro_auprc", "macro_auprc", "micro_auroc", "macro_auroc", "mrr",
            "brier", "ece", "hit_at_1", "recall_at_5", "recall_at_10")
    for key in keys:
        a, b = check.get(key), calibrated.get(key)
        if (a is None) != (b is None) or (a is not None and abs(float(a)-float(b)) > 5e-6):
            raise RuntimeError(f"{site}: regenerated calibrated point metric {key} differs from saved result")
    save_json(raw_status, {"status":"calculating_raw_full_test_metrics", "updated_utc":stamp()})
    raw = event_metric_report_with_subsample_ci(scores, targets, outcomes,
                bootstrap_repeats=200, bootstrap_seed=42, max_ci_rows=100_000)
    for key in ("site", "model", "split", "source_dataset", "patients_calibration", "patients_test",
                "patient_overlap", "calibration_rows", "test_rows", "train_rows_internal",
                "calibration_scope", "test_scope"):
        raw[key] = calibrated[key]
    raw.update(status="completed", calibration_state="before", temperature=1.0,
               fitted_temperature=calibrated["temperature"], metric_input="unmodified MAOMAO model logits",
               model_sha256=model_hash, calibrated_reference_sha256=reference_hash,
               same_test_targets_verified=True, calibrated_point_metrics_reproduced=True,
               created_utc=stamp())
    save_json(raw_path, raw)
    save_json(raw_status, {"status":"completed", "test_rows":raw["test_rows"], "finished_utc":stamp()})
    del scores, targets, check
    gc.collect()
    path.unlink(missing_ok=True)
    return raw


def evaluate(site: str, name: str, site_dir: Path, manifest: dict,
             outcomes: list[str], batch_size: int) -> dict:
    result_dir = RESULTS / site / name
    metrics_path = result_dir / "metrics.json"
    status_path = result_dir / "status.json"
    if metrics_path.exists() and status_path.exists():
        metrics = json.loads(metrics_path.read_text())
        status = json.loads(status_path.read_text())
        if (status.get("status") == "completed" and
                metrics.get("calibration_protocol") == PROTOCOL and
                metrics.get("test_rows") == manifest["test_target_rows"] and
                metrics.get("calibration_rows") == manifest["calibration_target_rows"] and
                (name != "xgboost" or metrics.get("model_sha256") == xgboost_revision())):
            if name == "maomao":
                save_maomao_uncalibrated(site, site_dir, result_dir, manifest, outcomes, batch_size, metrics)
            print(f"{site} {name}: existing complete metrics reused", flush=True)
            return metrics
    result_dir.mkdir(parents=True, exist_ok=True)
    status = {"site": site, "model": name, "status": "scoring_calibration",
              "started_utc": stamp(), "split": "patient-disjoint 90:10, seed 42"}
    save_json(status_path, status)
    calibration = make_logits(site, name, "calibration", site_dir, result_dir, batch_size)
    status["status"] = "scoring_test"
    save_json(status_path, status)
    test = make_logits(site, name, "test", site_dir, result_dir, batch_size)
    status["status"] = "fitting_temperature_on_all_calibration_rows"
    save_json(status_path, status)
    temperature, calibration_rows = fit_temperature(calibration, site_dir / "calibration_y.npy")
    status.update(status="evaluating_all_test_rows", temperature=temperature)
    save_json(status_path, status)
    scores = np.load(test, mmap_mode="r")
    targets = np.load(site_dir / "test_y.npy", mmap_mode="r")
    report = event_metric_report_with_subsample_ci(
        torch.from_numpy(np.array(scores, dtype=np.float32, copy=True)) / temperature,
        torch.from_numpy(np.array(targets, dtype=np.uint8, copy=True)), outcomes,
        bootstrap_repeats=200, bootstrap_seed=42, max_ci_rows=100_000)
    split = manifest["split"]
    report.update({
        "site": site, "model": name, "status": "completed",
        "split": "patient-disjoint 90:10, seed 42",
        "source_dataset": manifest["dataset"],
        "patients_calibration": split["patients_validation"],
        "patients_test": split["patients_test"],
        "patient_overlap": split["patient_overlap"],
        "calibration_rows": calibration_rows, "test_rows": int(len(targets)),
        "train_rows_internal": 14_128_539,
        "calibration_scope": "all valid target rows from the external 90% split",
        "test_scope": "all valid target rows from the external sealed 10% split",
        "temperature": temperature,
        "calibration_protocol": PROTOCOL,
        "metric_input": "model logits divided by external fitted temperature",
        "created_utc": stamp(),
    })
    if name == "xgboost":
        report["model_sha256"] = xgboost_revision()
    if name == "maomao":
        report.update(model_sha256=sha256(INTERNAL / "full_maomao_reference/best_model.pt"),
                      calibration_state="after")
    save_json(metrics_path, report)
    status.update(status="completed", finished_utc=stamp(),
                  calibration_rows=calibration_rows, test_rows=int(len(targets)),
                  metrics_file=str(metrics_path.relative_to(ROOT)))
    save_json(status_path, status)
    del scores, targets
    gc.collect()
    if name == "maomao":
        save_maomao_uncalibrated(site, site_dir, result_dir, manifest, outcomes, batch_size, report)
    calibration.unlink()
    test.unlink(missing_ok=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=tuple(SOURCES), required=True)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--maomao_batch_size", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(8)
    site_dir = BASE / args.site
    manifest = json.loads((site_dir / "manifest.json").read_text())
    if manifest.get("status") != "ready_for_full_scale_evaluation":
        raise RuntimeError(f"{args.site}: external rows are not complete")
    check_contract(ROOT / SOURCES[args.site], TRAINING)
    split = manifest["split"]
    if (split.get("patient_overlap") != 0 or
            split["patients_test"] != round(split["patients_total"] * 0.1) or
            split["patients_validation"] + split["patients_test"] != split["patients_total"]):
        raise RuntimeError(f"{args.site}: split is not patient-disjoint 90:10")
    for name, rows in (("calibration", manifest["calibration_target_rows"]),
                       ("test", manifest["test_target_rows"])):
        _, y = arrays(site_dir, name)
        if len(y) != rows:
            raise RuntimeError(f"{args.site}: {name} rows differ from manifest")
    outcomes = json.loads((TRAINING / "event_sequence_meta.json").read_text())["outcome_vocabulary"]
    summaries = {}
    for name in MODELS:
        existing = RESULTS / args.site / name / "metrics.json"
        if existing.exists():
            data = json.loads(existing.read_text())
            if data.get("test_rows") == manifest["test_target_rows"] and (name != "xgboost" or data.get("model_sha256") == xgboost_revision()):
                summaries[name] = {key:data[key] for key in ("micro_auprc", "micro_auroc", "test_rows")}
    for name in args.models:
        report = evaluate(args.site, name, site_dir, manifest, outcomes, args.maomao_batch_size)
        summaries[name] = {"micro_auprc": report["micro_auprc"],
                           "micro_auroc": report["micro_auroc"],
                           "test_rows": report["test_rows"]}
        save_json(RESULTS / args.site / "summary.json",
                  {"site": args.site, "status": "completed" if len(summaries) == len(MODELS) else "partial",
                   "patient_split": split, "models": summaries, "updated_utc": stamp()})
    print(json.dumps({"site": args.site, "models": list(summaries)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
