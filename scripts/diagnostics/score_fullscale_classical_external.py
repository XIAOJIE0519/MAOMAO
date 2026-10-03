#!/usr/bin/env python3
"""Score full-trained classical models on all external calibration/test rows."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from maomao.evaluation.event_metrics import event_metric_report_with_subsample_ci
from maomao.data.event_sequence import EventSequenceDataset, collate_event_sequences
from maomao.models.event_maomao import EventMAOMAO
from scripts.diagnostics.run_gru_baseline import GRUBaseline


def all_row_token_rates(x, y, num_tokens: int, chunk: int = 65536):
    global_counts = np.zeros(y.shape[1], dtype=np.float64)
    counts = np.zeros(num_tokens, dtype=np.float64)
    token_targets = np.zeros((num_tokens, y.shape[1]), dtype=np.float64)
    for start in range(0, len(x), chunk):
        end = min(len(x), start + chunk)
        ids = np.clip(np.rint(np.asarray(x[start:end, 0]) * (num_tokens - 1)).astype(np.int64), 0, num_tokens - 1)
        labels = np.asarray(y[start:end], dtype=np.uint8)
        global_counts += labels.sum(0)
        counts += np.bincount(ids, minlength=num_tokens)
        row_ids, col_ids = np.nonzero(labels)
        np.add.at(token_targets, (ids[row_ids], col_ids), 1.0)
        print(f"single-variable fit rows={end}/{len(x)}", flush=True)
    prior = (global_counts + 1.0) / (len(y) + 2.0)
    return (token_targets + prior[None, :]) / (counts[:, None] + 1.0)


def logistic_logits(x, checkpoint: dict, scaler: dict, device: torch.device,
                    chunk: int = 8192) -> np.ndarray:
    weight, bias = checkpoint["weight"], checkpoint["bias"]
    model = torch.nn.Linear(weight.shape[1], weight.shape[0]).to(device)
    model.weight.data.copy_(weight.to(device))
    model.bias.data.copy_(bias.to(device))
    mean = torch.as_tensor(scaler["mean"], device=device, dtype=torch.float32)
    scale = torch.as_tensor(scaler["scale"], device=device, dtype=torch.float32).clamp_min(1e-8)
    out = np.empty((len(x), weight.shape[0]), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), chunk):
            end = min(len(x), start + chunk)
            xb = torch.tensor(np.asarray(x[start:end], dtype=np.float32), device=device)
            xb.sub_(mean).div_(scale)
            out[start:end] = model(xb).float().cpu().numpy()
    return out


def metric_report(logits: np.ndarray, y: np.ndarray, outcomes: list[str],
                  name: str, repeats: int, seed: int) -> dict:
    report = event_metric_report_with_subsample_ci(
        torch.from_numpy(np.asarray(logits, dtype=np.float32)),
        torch.from_numpy(np.array(y, dtype=np.uint8, copy=True)), outcomes,
        bootstrap_repeats=repeats, bootstrap_seed=seed, max_ci_rows=100_000)
    report["model"] = name
    report["rows"] = int(len(y))
    return report


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    z = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    target = torch.from_numpy(np.array(y, dtype=np.float32, copy=True))
    log_t = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.Adam([log_t], lr=0.05)
    for _ in range(120):
        temperature = log_t.exp().clamp(0.15, 6.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z / temperature, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return float(log_t.detach().exp().clamp(0.15, 6.0))


def predict_sequence_checkpoint(kind: str, checkpoint: Path, data_dir: Path,
                                window_indices: np.ndarray, row_windows: np.ndarray,
                                row_positions: np.ndarray, output_dir: Path,
                                split_name: str, batch_size: int = 64) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = EventSequenceDataset(data_dir, 256, 128, dynamic_windows=False)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if kind == "maomao":
        cfg = payload["args"]
        model = EventMAOMAO(
            dataset.num_tokens, dataset.num_outcomes, dataset.num_static,
            cfg.get("hidden_dim", 384), cfg.get("num_layers", 10),
            cfg.get("num_heads", 12), cfg.get("ffn_dim", 1536),
            cfg.get("dropout", 0.1), cfg.get("initial_event_interval_hours", 24.0),
            cfg.get("decoupled_time_head", False), cfg.get("enhanced_time_encoding", False),
            len(dataset.trajectory_horizons_hours), cfg.get("lognormal_time_head", False),
            outcome_family_ids=torch.as_tensor(dataset.meta["outcome_to_family"], dtype=torch.long),
            use_family_head=cfg.get("use_family_head", True),
            same_time_block_causal=cfg.get("same_time_block_causal", False),
            relative_time_attention=cfg.get("relative_time_attention", False),
            event_conditioned_time_head=cfg.get("event_conditioned_time_head", False),
            phase_memory=cfg.get("phase_memory", False),
            clock_phase_context=cfg.get("clock_phase_context", True),
            observation_intensity=cfg.get("observation_intensity", False),
            value_reconstruction=cfg.get("masked_value_loss_weight", 0.0) > 0,
            dual_timescale_time_head=cfg.get("dual_timescale_time_head", False),
            fine_time_bins=cfg.get("fine_time_bins", 24),
            long_time_bins=cfg.get("long_time_bins", 44),
        ).to(device)
        clock_ids = [int(i) for token, i in dataset.meta["token_vocabulary"].items()
                     if token == "<CLOCK>" or token.startswith("phase_summary:")]
        model.clock_phase_token_ids = torch.as_tensor(clock_ids, dtype=torch.long, device=device)
    else:
        model = GRUBaseline(dataset.num_tokens, dataset.num_outcomes, dataset.num_static).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    rows_by_window: dict[int, list[tuple[int, int]]] = {}
    for row, (window, pos) in enumerate(zip(row_windows, row_positions)):
        rows_by_window.setdefault(int(window), []).append((row, int(pos)))
    logits_out = np.empty((len(row_windows), dataset.num_outcomes), dtype=np.float32)
    loader = DataLoader(Subset(dataset, window_indices), batch_size=batch_size,
                        shuffle=False, num_workers=0, collate_fn=collate_event_sequences)
    with torch.inference_mode():
        for n, batch in enumerate(loader, 1):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = model(batch)
            else:
                output = model(batch)
            logits = output.logits if kind == "maomao" else output
            batch_windows = window_indices[(n - 1) * batch_size:n * batch_size]
            for local, window in enumerate(batch_windows):
                for row, pos in rows_by_window.get(int(window), ()):
                    if not bool(batch["loss_mask"][local, pos]) or not bool(batch["target_set"][local, pos].sum() > 0):
                        raise RuntimeError(f"Invalid full external target row: window={window} position={pos}")
                    logits_out[row] = logits[local, pos].float().cpu().numpy()
            if n % 50 == 0:
                print(f"{kind} {split_name} batches={n}/{len(loader)}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"{kind}_{split_name}_logits.npy", logits_out)
    return logits_out


def main() -> None:
    base = ROOT / "outputs/final_experiment_results_20260923/classical_full_scale"
    metrics = ROOT / "outputs/final_experiment_results_20260923/baseline_metrics"
    metadata = json.loads((ROOT / "data/perioperative_event_sequences_v5_richctx_static7/event_sequence_meta.json").read_text())
    outcomes = metadata["outcome_vocabulary"]
    vocabulary = json.loads((ROOT / "data/perioperative_event_sequences_v5_richctx_static7/token_vocabulary.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logistic_checkpoint = torch.load(metrics / "logistic_fullscale.pt", map_location="cpu", weights_only=False)
    scaler_file = np.load(metrics / "logistic_scaler.npz")
    scaler = {"mean": scaler_file["mean"], "scale": scaler_file["scale"]}
    booster = None
    import xgboost as xgb
    booster = xgb.Booster()
    booster.load_model(str(metrics / "xgboost_fullscale.json"))
    internal_x = np.load(base / "train_X.npy", mmap_mode="r")
    internal_y = np.load(base / "train_y.npy", mmap_mode="r")
    token_rates = all_row_token_rates(internal_x, internal_y, len(vocabulary))

    result = {"protocol": "patient-disjoint 90% calibration / 10% sealed test; all valid rows",
              "models": {}}
    ext_base = base / "external"
    for site in ("mimic", "mover"):
        site_dir = ext_base / site
        manifest = json.loads((site_dir / "manifest.json").read_text())
        if manifest.get("status") != "ready_for_full_scale_evaluation":
            raise RuntimeError(f"External data for {site} is not fully materialized")
        x_cal = np.load(site_dir / "calibration_X.npy", mmap_mode="r")
        y_cal = np.load(site_dir / "calibration_y.npy", mmap_mode="r")
        x_test = np.load(site_dir / "test_X.npy", mmap_mode="r")
        y_test = np.load(site_dir / "test_y.npy", mmap_mode="r")
        cal_ids = np.clip(np.rint(np.asarray(x_cal[:, 0]) * (len(vocabulary) - 1)).astype(np.int64), 0, len(vocabulary) - 1)
        test_ids = np.clip(np.rint(np.asarray(x_test[:, 0]) * (len(vocabulary) - 1)).astype(np.int64), 0, len(vocabulary) - 1)
        models = {
            "univariate_last_token": (
                np.log(np.clip(token_rates[cal_ids], 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(token_rates[cal_ids], 1e-6, 1 - 1e-6)),
                np.log(np.clip(token_rates[test_ids], 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(token_rates[test_ids], 1e-6, 1 - 1e-6))),
            "logistic_regression": (logistic_logits(x_cal, logistic_checkpoint, scaler, device),
                                    logistic_logits(x_test, logistic_checkpoint, scaler, device)),
        }
        xgb_cal = xgb.DMatrix(np.asarray(x_cal, dtype=np.float32), nthread=16)
        xgb_test = xgb.DMatrix(np.asarray(x_test, dtype=np.float32), nthread=16)
        models["xgboost"] = (booster.predict(xgb_cal, output_margin=True),
                             booster.predict(xgb_test, output_margin=True))
        site_result = {"rows": int(len(y_test)), "calibration_rows": int(len(y_cal)),
                       "protocol": "patient-disjoint 90% calibration / 10% sealed test",
                       "models": {}, "calibrated_models": {}}
        for offset, (name, (cal_logits, test_logits)) in enumerate(models.items()):
            temperature = fit_temperature(cal_logits, y_cal)
            site_result["models"][name] = metric_report(test_logits, y_test, outcomes,
                                                        name, 200, 6200 + offset)
            calibrated = metric_report(test_logits / temperature, y_test, outcomes,
                                       name, 200, 6300 + offset)
            calibrated["temperature"] = temperature
            site_result["calibrated_models"][name] = calibrated
            print(f"external={site} model={name} calibration_rows={len(y_cal)} test_rows={len(y_test)} temperature={temperature:.4f}", flush=True)
        source_name = "val_mimic_richctx_static7" if site == "mimic" else "val_mover_richctx_static7"
        source_dir = ROOT / "data" / source_name
        for offset, (name, kind, checkpoint) in enumerate((
            ("gru", "gru", ROOT / "outputs/baseline_gru_richctx/gru_best.pt"),
            ("maomao", "maomao", ROOT / "outputs/final_experiment_results_20260923/full_maomao_reference/best_model.pt"),
        ), start=10):
            cal_logits = predict_sequence_checkpoint(
                kind, checkpoint, source_dir,
                np.load(site_dir / "calibration_window_indices.npy", mmap_mode="r"),
                np.load(site_dir / "calibration_window_indices.npy", mmap_mode="r"),
                np.load(site_dir / "calibration_positions.npy", mmap_mode="r"),
                site_dir, "calibration")
            test_logits = predict_sequence_checkpoint(
                kind, checkpoint, source_dir,
                np.load(site_dir / "test_window_indices.npy", mmap_mode="r"),
                np.load(site_dir / "test_window_indices.npy", mmap_mode="r"),
                np.load(site_dir / "test_positions.npy", mmap_mode="r"),
                site_dir, "test")
            temperature = fit_temperature(cal_logits, y_cal)
            site_result["models"][name] = metric_report(test_logits, y_test, outcomes,
                                                        name, 200, 6400 + offset)
            calibrated = metric_report(test_logits / temperature, y_test, outcomes,
                                       name, 200, 6500 + offset)
            calibrated["temperature"] = temperature
            site_result["calibrated_models"][name] = calibrated
            print(f"external={site} model={name} all_rows={len(y_test)} temperature={temperature:.4f}", flush=True)
        result[site] = site_result
    (metrics / "fullscale_classical_external.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
