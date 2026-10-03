#!/usr/bin/env python3
"""Subject-ID-grouped external validation, calibration, and final test scoring.

The source event artifacts are never rewritten. For each source, operation
episodes are grouped by patient, a configurable fraction is sealed as a test
split, and the remaining patients are used to select lightweight output calibration. The
transformer weights stay frozen: calibration can therefore be updated for a
new hospital without erasing the perioperative representation learned from
the full training cohort.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from maomao.data.event_sequence import EventSequenceDataset, collate_event_sequences  # noqa: E402
from maomao.models.event_maomao import EventMAOMAO  # noqa: E402
from maomao.data.outcome_families import family_hit_counts, family_ids  # noqa: E402
from maomao.evaluation.event_metrics import event_metric_report  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("outputs/event_maomao_v5_full/best_model.pt"))
    parser.add_argument("--training_dir", type=Path,
                        default=Path("data/perioperative_event_sequences_v5_full"))
    parser.add_argument("--data_dirs", nargs="+", type=Path,
                        default=[Path("data/val_mimic"), Path("data/val_mover")])
    parser.add_argument("--output_dir", type=Path,
                        default=Path("outputs/external_validation_v5"))
    parser.add_argument("--test_fraction", type=float, default=0.30,
                        help="Sealed test fraction; default 30%% leaves 70%% for calibration")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_calibration_targets", type=int, default=200_000)
    parser.add_argument("--expanded_metric_limit", type=int, default=100_000,
                        help="Uniform reservoir sample size for expanded AUROC/AUPRC metrics; 0 disables them")
    parser.add_argument("--skip_validation_scores", action="store_true",
                        help="Fit calibration on validation windows but skip duplicate raw/calibrated validation scoring")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vocabulary_size", type=int, choices=(0, 50, 100, 150), default=0,
                        help="Runtime event vocabulary limit for vocabulary ablations; 0 uses Full")
    return parser.parse_args()


def json_dump(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    temporary.replace(path)


def check_contract(data_dir: Path, training_dir: Path) -> dict:
    meta = json.loads((data_dir / "event_sequence_meta.json").read_text())
    train = json.loads((training_dir / "event_sequence_meta.json").read_text())
    keys = ("token_vocabulary", "outcome_vocabulary", "stored_outcome_vocabulary",
            "outcome_class_remap", "num_static", "trajectory_horizons_hours")
    mismatches = [key for key in keys if meta.get(key) != train.get(key)]
    if mismatches:
        raise ValueError(f"{data_dir}: frozen training contract mismatch: {mismatches}")
    if meta.get("split") != "external_validation" or not meta.get("complete"):
        raise ValueError(f"{data_dir}: expected a complete external_validation artifact")
    return {
        "compatible": True,
        "version": meta.get("version"),
        "source_name": meta.get("source_name"),
        "vocabulary_size": len(meta["token_vocabulary"]),
        "outcome_classes": len(meta["outcome_vocabulary"]),
        "static_features": meta.get("static_features"),
        "trajectory_horizons_hours": meta.get("trajectory_horizons_hours"),
    }


def patient_split(dataset: EventSequenceDataset, data_dir: Path, output_dir: Path,
                  test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    admissions = pd.read_csv(data_dir / "admissions.csv", dtype={"subject_id": "string"})
    if len(admissions) != len(dataset.ptr) - 1:
        raise ValueError(f"{data_dir}: admissions.csv is not aligned with sequence_ptr.npy")
    group_column = "subject_id" if "subject_id" in admissions else "admission_id"
    groups = admissions[group_column].fillna("missing:").astype(str).to_numpy()
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = unique_groups.copy()
    rng.shuffle(shuffled)
    test_group_count = max(1, int(round(len(shuffled) * test_fraction)))
    test_groups = set(shuffled[:test_group_count].tolist())
    test_admission_mask = np.fromiter(
        (group in test_groups for group in groups), dtype=bool, count=len(groups))
    test_admissions = np.flatnonzero(test_admission_mask).astype(np.int32)
    validation_admissions = np.flatnonzero(~test_admission_mask).astype(np.int32)
    if np.intersect1d(validation_admissions, test_admissions).size:
        raise AssertionError("admission leakage between validation and test")
    validation_windows = np.flatnonzero(
        np.isin(dataset.window_admission, validation_admissions)).astype(np.int64)
    test_windows = np.flatnonzero(
        np.isin(dataset.window_admission, test_admissions)).astype(np.int64)
    if np.intersect1d(validation_windows, test_windows).size:
        raise AssertionError("window leakage between validation and test")

    split_path = output_dir / f"{data_dir.name}_patient_split.npz"
    np.savez_compressed(
        split_path,
        validation_admissions=validation_admissions,
        test_admissions=test_admissions,
        validation_windows=validation_windows,
        test_windows=test_windows,
    )
    summary = {
        "seed": seed,
        "group_column": group_column,
        "patients_total": int(len(unique_groups)),
        "patients_validation": int(len(unique_groups) - test_group_count),
        "patients_test": int(test_group_count),
        "episodes_total": int(len(groups)),
        "episodes_validation": int(len(validation_admissions)),
        "episodes_test": int(len(test_admissions)),
        "episode_test_fraction": float(len(test_admissions) / len(groups)),
        "windows_validation": int(len(validation_windows)),
        "windows_test": int(len(test_windows)),
        "patient_overlap": 0,
        "split_file": str(split_path.resolve()),
    }
    return validation_windows, test_windows, summary


def make_loader(dataset, indices: Iterable[int], batch_size: int, workers: int,
                device: torch.device) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False,
        num_workers=workers, persistent_workers=workers > 0,
        pin_memory=device.type == "cuda", collate_fn=collate_event_sequences,
    )


def build_model(dataset, checkpoint: dict, device: torch.device) -> EventMAOMAO:
    config = checkpoint["args"]
    has_hierarchical_head = any(
        key.startswith("family_head.") for key in checkpoint["model"])
    outcome_family_ids = torch.as_tensor(
        dataset.meta.get("outcome_to_family", family_ids(
            dataset.meta["outcome_vocabulary"]).tolist()), dtype=torch.long)
    model = EventMAOMAO(
        dataset.num_tokens, dataset.num_outcomes, dataset.num_static,
        config.get("hidden_dim", 256), config.get("num_layers", 8),
        config.get("num_heads", 8), config.get("ffn_dim", 1024),
        config.get("dropout", 0.1), config.get("initial_event_interval_hours", 24.0),
        config.get("decoupled_time_head", False),
        config.get("enhanced_time_encoding", False),
        len(dataset.trajectory_horizons_hours),
        config.get("lognormal_time_head", False),
        outcome_family_ids=outcome_family_ids,
        use_family_head=config.get("use_family_head", has_hierarchical_head),
        same_time_block_causal=config.get("same_time_block_causal", False),
        relative_time_attention=config.get("relative_time_attention", False),
        event_conditioned_time_head=config.get("event_conditioned_time_head", False),
        phase_memory=config.get("phase_memory", False),
        clock_phase_context=config.get("clock_phase_context", True),
        observation_intensity=config.get("observation_intensity", False),
        value_reconstruction=config.get("masked_value_loss_weight", 0.0) > 0,
        dual_timescale_time_head=config.get("dual_timescale_time_head", False),
        fine_time_bins=config.get("fine_time_bins", 24),
        long_time_bins=config.get("long_time_bins", 44),
    ).to(device)
    clock_phase_ids = [int(token_id) for token, token_id in dataset.meta["token_vocabulary"].items()
                       if token == "<CLOCK>" or token.startswith("phase_summary:")]
    model.clock_phase_token_ids = torch.as_tensor(clock_phase_ids, dtype=torch.long, device=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


@torch.no_grad()
def collect_calibration(model, dataset, window_indices: np.ndarray, device: torch.device,
                        batch_size: int, workers: int, max_targets: int,
                        amp_dtype: torch.dtype, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    order = window_indices.copy()
    rng.shuffle(order)
    loader = make_loader(dataset, order, batch_size, workers, device)
    event_logits, event_targets = [], []
    time_mu, time_log_sigma, time_dt, time_is_event = [], [], [], []
    event_count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=device.type == "cuda"):
            output = model(batch)
        event_valid = batch["loss_mask"].bool() & (batch["target_set"].sum(-1) > 0)
        time_valid = batch["time_mask"].bool()
        if event_valid.any():
            event_logits.append(output.logits[event_valid].float().cpu())
            event_targets.append(batch["target_set"][event_valid].bool().cpu())
            event_count += int(event_valid.sum())
        if time_valid.any():
            time_mu.append(output.time_mu[time_valid].float().cpu())
            time_log_sigma.append(output.time_log_sigma[time_valid].float().cpu())
            time_dt.append(batch["target_dt_hours"][time_valid].float().cpu())
            time_is_event.append(event_valid[time_valid].cpu())
        if event_count >= max_targets:
            break
    result = {
        "event_logits": torch.cat(event_logits)[:max_targets],
        "event_targets": torch.cat(event_targets)[:max_targets],
        "time_mu": torch.cat(time_mu)[:max_targets],
        "time_log_sigma": torch.cat(time_log_sigma)[:max_targets],
        "time_dt": torch.cat(time_dt)[:max_targets],
        "time_is_event": torch.cat(time_is_event)[:max_targets],
    }
    result["sampled_event_targets"] = len(result["event_logits"])
    result["sampled_time_targets"] = len(result["time_mu"])
    return result


def event_nll(logits: torch.Tensor, targets: torch.Tensor,
              log_temperature: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    calibrated = (logits + bias) / log_temperature.exp().clamp(0.15, 6.0)
    positive = calibrated.masked_fill(~targets, -torch.inf)
    return (torch.logsumexp(calibrated, -1) - torch.logsumexp(positive, -1)).mean()


def fit_event_family(logits: torch.Tensor, targets: torch.Tensor, family: str,
                     steps: int = 180) -> dict:
    device = logits.device
    learn_bias = family == "temperature_and_bias"
    log_temperature = torch.zeros((), device=device, requires_grad=True)
    bias = torch.zeros(logits.shape[1], device=device, requires_grad=learn_bias)
    parameters = [log_temperature] + ([bias] if learn_bias else [])
    optimizer = torch.optim.Adam(parameters, lr=0.04 if learn_bias else 0.025)
    active = targets.any(0).float()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        effective_bias = bias * active
        loss = event_nll(logits, targets, log_temperature, effective_bias)
        if learn_bias:
            loss = loss + 2e-3 * effective_bias.square().mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            log_temperature.clamp_(-1.9, 1.8)
            if learn_bias:
                bias.clamp_(-4.0, 4.0)
                if active.sum():
                    bias.sub_((bias * active).sum() / active.sum())
    return {
        "family": family,
        "temperature": float(log_temperature.detach().exp()),
        "bias": (bias.detach() * active).cpu().tolist(),
    }


def apply_event(logits: torch.Tensor, calibration: dict) -> torch.Tensor:
    bias = torch.as_tensor(calibration["bias"], device=logits.device, dtype=logits.dtype)
    return (logits + bias) / float(calibration["temperature"])


def time_nll(mu: torch.Tensor, log_sigma: torch.Tensor, dt: torch.Tensor,
             is_event: torch.Tensor, mu_shift: torch.Tensor,
             sigma_shift: torch.Tensor) -> torch.Tensor:
    adjusted_mu = mu + mu_shift
    adjusted_log_sigma = (log_sigma + sigma_shift).clamp(-3.0, 2.0)
    sigma = adjusted_log_sigma.exp()
    log_dt = dt.clamp_min(1.0 / 60.0).log()
    z = (log_dt - adjusted_mu) / sigma
    event_loss = 0.5 * z.square() + adjusted_log_sigma + log_dt + 0.5 * math.log(2 * math.pi)
    survival = (0.5 * torch.erfc(z / math.sqrt(2.0))).clamp_min(1e-7)
    return torch.where(is_event, event_loss, -survival.log()).mean()


def fit_time_family(mu, log_sigma, dt, is_event, family: str, steps: int = 160) -> dict:
    learn_mu = family in {"mu_shift", "mu_and_sigma_shift"}
    learn_sigma = family == "mu_and_sigma_shift"
    mu_shift = torch.zeros((), device=mu.device, requires_grad=learn_mu)
    sigma_shift = torch.zeros((), device=mu.device, requires_grad=learn_sigma)
    parameters = ([mu_shift] if learn_mu else []) + ([sigma_shift] if learn_sigma else [])
    if parameters:
        optimizer = torch.optim.Adam(parameters, lr=0.025)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = time_nll(mu, log_sigma, dt, is_event, mu_shift, sigma_shift)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                mu_shift.clamp_(-3.0, 3.0)
                sigma_shift.clamp_(-1.5, 1.5)
    return {
        "family": family,
        "mu_shift": float(mu_shift.detach()),
        "log_sigma_shift": float(sigma_shift.detach()),
    }


def point_prediction(mu: torch.Tensor, log_sigma: torch.Tensor,
                     family: str, log_scale: float,
                     max_wait_hours: float = 24.0 * 38) -> torch.Tensor:
    sigma = log_sigma.clamp(-3.0, 2.0).exp()
    if family == "scaled_median":
        base = torch.exp(mu)
    else:
        base = torch.exp(mu + 0.5 * sigma.square())
    return (base * math.exp(float(log_scale))).clamp_max(max_wait_hours)


def fit_point_scale(mu: torch.Tensor, log_sigma: torch.Tensor, dt: torch.Tensor,
                    family: str) -> dict:
    """Fit a robust one-parameter point forecast without changing uncertainty."""
    grid = torch.linspace(-3.0, 3.0, 241, device=mu.device)
    base_family = "scaled_median" if family == "scaled_median" else "scaled_mean"
    sigma = log_sigma.clamp(-3.0, 2.0).exp()
    base = torch.exp(mu) if base_family == "scaled_median" else torch.exp(
        mu + 0.5 * sigma.square())
    # Chunk the grid to keep the temporary error tensor bounded.
    scores = []
    for start in range(0, len(grid), 32):
        scales = grid[start:start + 32].exp()[:, None]
        prediction = (scales * base[None, :]).clamp_max(24.0 * 38)
        scores.append((prediction - dt[None, :]).abs().mean(1))
    scores = torch.cat(scores)
    best = int(scores.argmin())
    return {"point_family": base_family, "point_log_scale": float(grid[best])}


def select_calibration(collected: dict, device: torch.device, seed: int) -> tuple[dict, dict]:
    logits = collected["event_logits"].to(device)
    targets = collected["event_targets"].to(device)
    mu = collected["time_mu"].to(device)
    log_sigma = collected["time_log_sigma"].to(device)
    dt = collected["time_dt"].to(device)
    is_event = collected["time_is_event"].to(device)
    generator = torch.Generator().manual_seed(seed)
    event_order = torch.randperm(len(logits), generator=generator)
    time_order = torch.randperm(len(mu), generator=generator)
    event_cut = max(1, int(0.8 * len(event_order)))
    time_cut = max(1, int(0.8 * len(time_order)))
    event_fit, event_select = event_order[:event_cut].to(device), event_order[event_cut:].to(device)
    time_fit, time_select = time_order[:time_cut].to(device), time_order[time_cut:].to(device)

    event_trials = []
    zero_bias = [0.0] * logits.shape[1]
    raw_event = {"family": "raw", "temperature": 1.0, "bias": zero_bias}
    for candidate in [raw_event] + [
        fit_event_family(logits[event_fit], targets[event_fit], "temperature_only"),
        fit_event_family(logits[event_fit], targets[event_fit], "temperature_and_bias"),
    ]:
        score = event_nll(
            logits[event_select], targets[event_select],
            torch.tensor(math.log(candidate["temperature"]), device=device),
            torch.tensor(candidate["bias"], device=device),
        )
        event_trials.append({**candidate, "selection_nll": float(score)})
    selected_event_family = min(event_trials, key=lambda row: row["selection_nll"])["family"]
    if selected_event_family == "raw":
        final_event = raw_event
    else:
        final_event = fit_event_family(logits, targets, selected_event_family, steps=220)

    raw_time = {"family": "raw", "mu_shift": 0.0, "log_sigma_shift": 0.0}
    time_trials = []
    for candidate in [raw_time] + [
        fit_time_family(mu[time_fit], log_sigma[time_fit], dt[time_fit],
                        is_event[time_fit], "mu_shift"),
        fit_time_family(mu[time_fit], log_sigma[time_fit], dt[time_fit],
                        is_event[time_fit], "mu_and_sigma_shift"),
    ]:
        score = time_nll(
            mu[time_select], log_sigma[time_select], dt[time_select], is_event[time_select],
            torch.tensor(candidate["mu_shift"], device=device),
            torch.tensor(candidate["log_sigma_shift"], device=device),
        )
        time_trials.append({**candidate, "selection_nll": float(score)})
    selected_time_family = min(time_trials, key=lambda row: row["selection_nll"])["family"]
    if selected_time_family == "raw":
        final_time = raw_time
    else:
        final_time = fit_time_family(mu, log_sigma, dt, is_event,
                                     selected_time_family, steps=200)

    # Censor-aware likelihood and a clinically displayed point forecast serve
    # different purposes.  Select the point estimator only on observed events
    # and keep it independent of the distributional sigma calibration.
    fit_observed = time_fit[is_event[time_fit]]
    select_observed = time_select[is_event[time_select]]
    point_trials = []
    point_candidates = [
        {"point_family": "raw_mean", "point_log_scale": 0.0},
        fit_point_scale(mu[fit_observed], log_sigma[fit_observed], dt[fit_observed],
                        "scaled_mean"),
        fit_point_scale(mu[fit_observed], log_sigma[fit_observed], dt[fit_observed],
                        "scaled_median"),
    ]
    for candidate in point_candidates:
        prediction = point_prediction(
            mu[select_observed], log_sigma[select_observed],
            candidate["point_family"], candidate["point_log_scale"])
        mae = (prediction - dt[select_observed]).abs().mean()
        point_trials.append({**candidate, "selection_mae_hours": float(mae)})
    selected_point_family = min(
        point_trials, key=lambda row: row["selection_mae_hours"])["point_family"]
    if selected_point_family == "raw_mean":
        final_point = {"point_family": "raw_mean", "point_log_scale": 0.0}
    else:
        all_observed = is_event.nonzero(as_tuple=True)[0]
        final_point = fit_point_scale(
            mu[all_observed], log_sigma[all_observed], dt[all_observed],
            selected_point_family)
    final_time.update(final_point)
    selected = {"event": final_event, "time": final_time}
    trials = {"event": event_trials, "time_distribution": time_trials,
              "time_point": point_trials}
    return selected, trials


@torch.no_grad()
def evaluate(model, loader, device: torch.device, amp_dtype: torch.dtype,
             calibration: dict, max_wait_hours: float,
             outcome_names: list[str], core_outcomes: set[str] | None = None,
             expanded_metric_limit: int = 100000,
             metric_seed: int = 42) -> dict:
    totals = {
        "event_targets": 0, "event_nll_sum": 0.0, "hit5": 0, "hit10": 0,
        "hit_all": 0, "family_hit5": 0, "family_hit10": 0,
        "family_hit_all": 0,
        "time_targets": 0, "time_events": 0, "time_nll_sum": 0.0,
        "time_abs_error_sum": 0.0, "time_squared_error_sum": 0.0,
        "trajectory_values": 0, "trajectory_bce_sum": 0.0,
        "trajectory_brier_sum": 0.0,
    }
    event_cal = calibration["event"]
    time_cal = calibration["time"]
    class_positive = torch.zeros(len(outcome_names), dtype=torch.long)
    class_hit5 = torch.zeros(len(outcome_names), dtype=torch.long)
    class_hit10 = torch.zeros(len(outcome_names), dtype=torch.long)
    outcome_family_ids = family_ids(outcome_names, device)
    # Expanded ranking metrics are expensive over millions of event targets.
    # Keep a deterministic uniform reservoir over the *entire* evaluation
    # split, rather than taking the first N rows (which can overrepresent
    # early records or hospitals in source-sorted datasets).
    metric_rng = np.random.default_rng(metric_seed)
    metric_logit_chunks: list[np.ndarray] = []
    metric_target_chunks: list[np.ndarray] = []
    metric_priority_chunks: list[np.ndarray] = []
    metric_buffer_rows = 0

    def trim_metric_reservoir(limit: int) -> None:
        nonlocal metric_logit_chunks, metric_target_chunks
        nonlocal metric_priority_chunks, metric_buffer_rows
        if metric_buffer_rows <= limit:
            return
        all_logits = np.concatenate(metric_logit_chunks, axis=0)
        all_targets = np.concatenate(metric_target_chunks, axis=0)
        all_priorities = np.concatenate(metric_priority_chunks, axis=0)
        keep = np.argpartition(all_priorities, -limit)[-limit:]
        metric_logit_chunks = [all_logits[keep].copy()]
        metric_target_chunks = [all_targets[keep].copy()]
        metric_priority_chunks = [all_priorities[keep].copy()]
        metric_buffer_rows = limit
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=device.type == "cuda"):
            output = model(batch)
        event_valid = batch["loss_mask"].bool() & (batch["target_set"].sum(-1) > 0)
        time_valid = batch["time_mask"].bool()
        if event_valid.any():
            logits = apply_event(output.logits[event_valid].float(), event_cal)
            targets = batch["target_set"][event_valid].bool()
            positive = logits.masked_fill(~targets, -torch.inf)
            nll = torch.logsumexp(logits, -1) - torch.logsumexp(positive, -1)
            top = logits.topk(min(10, logits.shape[-1]), dim=-1).indices
            top5 = top[:, :min(5, top.shape[1])]
            top10 = top[:, :min(10, top.shape[1])]
            predicted5 = torch.zeros_like(targets).scatter_(1, top5, True)
            predicted10 = torch.zeros_like(targets).scatter_(1, top10, True)
            totals["event_targets"] += len(logits)
            totals["event_nll_sum"] += float(nll.sum())
            totals["hit5"] += int((targets & predicted5).any(1).sum())
            totals["hit10"] += int((targets & predicted10).any(1).sum())
            totals["hit_all"] += int((~targets | predicted10).all(1).sum())
            family_hit5, _ = family_hit_counts(targets, top5, outcome_family_ids)
            family_hit10, family_hit_all = family_hit_counts(
                targets, top10, outcome_family_ids)
            totals["family_hit5"] += family_hit5
            totals["family_hit10"] += family_hit10
            totals["family_hit_all"] += family_hit_all
            class_positive += targets.sum(0).long().cpu()
            class_hit5 += (targets & predicted5).sum(0).long().cpu()
            class_hit10 += (targets & predicted10).sum(0).long().cpu()
            if expanded_metric_limit > 0:
                metric_logit_chunks.append(logits.cpu().numpy())
                metric_target_chunks.append(targets.cpu().numpy())
                metric_priority_chunks.append(metric_rng.random(len(logits)))
                metric_buffer_rows += len(logits)
                if metric_buffer_rows >= 2 * expanded_metric_limit:
                    trim_metric_reservoir(expanded_metric_limit)
        if time_valid.any():
            mu = output.time_mu[time_valid].float() + float(time_cal["mu_shift"])
            log_sigma = (output.time_log_sigma[time_valid].float() +
                         float(time_cal["log_sigma_shift"])).clamp(-3.0, 2.0)
            dt = batch["target_dt_hours"][time_valid].float().clamp(
                1.0 / 60.0, max_wait_hours)
            observed = event_valid[time_valid]
            nll = time_nll(mu, log_sigma, dt, observed,
                           torch.zeros((), device=device), torch.zeros((), device=device))
            totals["time_targets"] += len(dt)
            totals["time_nll_sum"] += float(nll) * len(dt)
            if observed.any():
                # Point calibration is deliberately based on the original
                # model distribution, not the censor-likelihood sigma shift.
                raw_mu = output.time_mu[time_valid].float()
                raw_log_sigma = output.time_log_sigma[time_valid].float()
                expected = point_prediction(
                    raw_mu, raw_log_sigma,
                    time_cal.get("point_family", "raw_mean"),
                    time_cal.get("point_log_scale", 0.0), max_wait_hours)
                error = expected[observed] - dt[observed]
                totals["time_events"] += int(observed.sum())
                totals["time_abs_error_sum"] += float(error.abs().sum())
                totals["time_squared_error_sum"] += float(error.square().sum())
        trajectory_valid = batch["trajectory_mask"].bool().unsqueeze(-1).expand_as(
            batch["trajectory_target"])
        if trajectory_valid.any():
            trajectory_logits = output.trajectory_logits[trajectory_valid].float()
            trajectory_targets = batch["trajectory_target"][trajectory_valid].float()
            totals["trajectory_values"] += len(trajectory_logits)
            totals["trajectory_bce_sum"] += float(F.binary_cross_entropy_with_logits(
                trajectory_logits, trajectory_targets, reduction="sum"))
            totals["trajectory_brier_sum"] += float(
                (trajectory_logits.sigmoid() - trajectory_targets).square().sum())
    event_n = max(1, totals["event_targets"])
    time_n = max(1, totals["time_targets"])
    observed_n = max(1, totals["time_events"])
    trajectory_n = max(1, totals["trajectory_values"])
    report = {
        "event_targets": totals["event_targets"],
        "event_set_nll": totals["event_nll_sum"] / event_n,
        "next_event_hit_at_5": totals["hit5"] / event_n,
        "next_event_hit_at_10": totals["hit10"] / event_n,
        "all_true_events_hit_at_10": totals["hit_all"] / event_n,
        "same_family_hit_at_5": totals["family_hit5"] / event_n,
        "same_family_hit_at_10": totals["family_hit10"] / event_n,
        "all_true_families_hit_at_10": totals["family_hit_all"] / event_n,
        "all_outcome_support": {
            name: int(class_positive[index])
            for index, name in enumerate(outcome_names)
        },
        "all_outcome_hit_at_5": {
            name: (float(class_hit5[index] / class_positive[index])
                   if class_positive[index] else None)
            for index, name in enumerate(outcome_names)
        },
        "all_outcome_hit_at_10": {
            name: (float(class_hit10[index] / class_positive[index])
                   if class_positive[index] else None)
            for index, name in enumerate(outcome_names)
        },
        "time_targets_including_censoring": totals["time_targets"],
        "observed_time_targets": totals["time_events"],
        "time_nll": totals["time_nll_sum"] / time_n,
        "time_mae_hours": totals["time_abs_error_sum"] / observed_n,
        "time_rmse_hours": math.sqrt(totals["time_squared_error_sum"] / observed_n),
        "trajectory_bce": totals["trajectory_bce_sum"] / trajectory_n,
        "trajectory_brier": totals["trajectory_brier_sum"] / trajectory_n,
    }
    if metric_buffer_rows:
        trim_metric_reservoir(expanded_metric_limit)
        sampled_logits = torch.from_numpy(np.concatenate(metric_logit_chunks, axis=0))
        sampled_targets = torch.from_numpy(np.concatenate(metric_target_chunks, axis=0))
        report["expanded_metrics_full"] = event_metric_report(
            sampled_logits, sampled_targets, outcome_names)
        report["expanded_metrics_sampled_targets"] = len(sampled_logits)
        if core_outcomes:
            core_mask = torch.tensor([name in core_outcomes for name in outcome_names])
            report["expanded_metrics_core"] = event_metric_report(
                sampled_logits, sampled_targets, outcome_names, core_mask)
            report["core_outcomes"] = [name for name in outcome_names if name in core_outcomes]
    return report


def main():
    args = parse_args()
    if not 0 < args.test_fraction <= 0.5:
        raise ValueError("test_fraction must be greater than 0 and at most 0.5")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full external evaluation")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    amp_dtype = (torch.bfloat16 if checkpoint["args"].get("precision", "bf16") == "bf16"
                 else torch.float16)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Vocabulary ablations must preserve the training checkpoint's class order.
    # Re-ranking by external-site frequency changes both outcome semantics and
    # the number/order of event families, which makes the checkpoint unloadable
    # and would otherwise silently invalidate the comparison.
    runtime_outcome_names = None
    if args.vocabulary_size:
        reference = EventSequenceDataset(
            args.training_dir, checkpoint["args"].get("block_size", 256),
            checkpoint["args"].get("window_stride", 128),
            outcome_limit=args.vocabulary_size)
        runtime_outcome_names = list(reference.meta["outcome_vocabulary"])
    external_metas = [json.loads((path / "event_sequence_meta.json").read_text())
                      for path in args.data_dirs]
    supported_sets = []
    for meta in external_metas:
        counts = meta.get("audit_counts", {})
        supported_sets.append({name for name in meta["outcome_vocabulary"]
                               if int(counts.get(name, 0)) > 0})
    core_outcomes = set.intersection(*supported_sets) if supported_sets else set()
    reports = []
    for data_dir in args.data_dirs:
        print(f"[{data_dir.name}] checking frozen contract", flush=True)
        contract = check_contract(data_dir, args.training_dir)
        dataset = EventSequenceDataset(
            data_dir, checkpoint["args"].get("block_size", 256),
            checkpoint["args"].get("window_stride", 128),
            outcome_limit=args.vocabulary_size,
            outcome_names=runtime_outcome_names)
        validation_windows, test_windows, split = patient_split(
            dataset, data_dir, args.output_dir, args.test_fraction, args.seed)
        model = build_model(dataset, checkpoint, device)
        print(f"[{data_dir.name}] collecting validation calibration targets", flush=True)
        collected = collect_calibration(
            model, dataset, validation_windows, device, args.batch_size,
            args.num_workers, args.max_calibration_targets, amp_dtype, args.seed)
        calibration, trials = select_calibration(collected, device, args.seed)
        validation_percent = 100.0 * (1.0 - args.test_fraction)
        test_percent = 100.0 * args.test_fraction
        raw = {"event": {"family": "raw", "temperature": 1.0,
                         "bias": [0.0] * dataset.num_outcomes},
               "time": {"family": "raw", "mu_shift": 0.0, "log_sigma_shift": 0.0,
                        "point_family": "raw_mean", "point_log_scale": 0.0}}
        if args.skip_validation_scores:
            print(f"[{data_dir.name}] validation calibration frozen; skipping duplicate validation scoring",
                  flush=True)
            validation_raw = {"evaluation_skipped": True}
            validation_calibrated = {"evaluation_skipped": True}
        else:
            validation_loader = make_loader(
                dataset, validation_windows, args.batch_size, args.num_workers, device)
            print(f"[{data_dir.name}] scoring {validation_percent:g}% validation split", flush=True)
            validation_raw = evaluate(
                model, validation_loader, device, amp_dtype, raw,
                checkpoint["args"].get("max_wait_hours", 912.0),
                dataset.meta["outcome_vocabulary"], core_outcomes,
                args.expanded_metric_limit, args.seed + 9001)
            validation_calibrated = evaluate(
                model, validation_loader, device, amp_dtype, calibration,
                checkpoint["args"].get("max_wait_hours", 912.0),
                dataset.meta["outcome_vocabulary"], core_outcomes,
                args.expanded_metric_limit, args.seed + 9001)
            del validation_loader
        test_loader = make_loader(
            dataset, test_windows, args.batch_size, args.num_workers, device)
        print(f"[{data_dir.name}] calibration frozen; scoring sealed "
              f"{test_percent:g}% test split", flush=True)
        test_raw = evaluate(
            model, test_loader, device, amp_dtype, raw,
            checkpoint["args"].get("max_wait_hours", 912.0),
            dataset.meta["outcome_vocabulary"], core_outcomes,
            args.expanded_metric_limit, args.seed + 9001)
        test_calibrated = evaluate(
            model, test_loader, device, amp_dtype, calibration,
            checkpoint["args"].get("max_wait_hours", 912.0),
            dataset.meta["outcome_vocabulary"], core_outcomes,
            args.expanded_metric_limit, args.seed + 9001)
        report = {
            "source": data_dir.name,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_step": int(checkpoint["global_step"]),
            "contract": contract,
            "split": split,
            "calibration_sample": {
                "event_targets": collected["sampled_event_targets"],
                "time_targets": collected["sampled_time_targets"],
            },
            "expanded_metric_policy": {
                "max_targets": args.expanded_metric_limit,
                "sampling": "deterministic uniform reservoir over the full scored split",
                "seed": args.seed + 9001,
            },
            "calibration_trials": trials,
            "selected_calibration": calibration,
            "validation_split": {
                "raw": validation_raw, "calibrated": validation_calibrated},
            "test_split_final": {
                "raw": test_raw, "calibrated": test_calibrated},
        }
        report_path = args.output_dir / f"{data_dir.name}_evaluation.json"
        json_dump(report, report_path)
        outcome_rows = []
        raw_per_event = test_raw.get("expanded_metrics_full", {}).get("per_event", {})
        calibrated_per_event = test_calibrated.get("expanded_metrics_full", {}).get("per_event", {})
        for outcome_name in dataset.meta["outcome_vocabulary"]:
            raw_metrics = raw_per_event.get(outcome_name, {})
            calibrated_metrics = calibrated_per_event.get(outcome_name, {})
            outcome_rows.append({
                "outcome": outcome_name,
                "test_support": test_raw["all_outcome_support"][outcome_name],
                "metric_sample_support_raw": raw_metrics.get("support"),
                "metric_sample_support_calibrated": calibrated_metrics.get("support"),
                "raw_auroc_sampled": raw_metrics.get("auroc"),
                "calibrated_auroc_sampled": calibrated_metrics.get("auroc"),
                "raw_auprc_sampled": raw_metrics.get("auprc"),
                "calibrated_auprc_sampled": calibrated_metrics.get("auprc"),
                "raw_recall_at_5_sampled": raw_metrics.get("recall_at_5"),
                "calibrated_recall_at_5_sampled": calibrated_metrics.get("recall_at_5"),
                "raw_hit_at_5": test_raw["all_outcome_hit_at_5"][outcome_name],
                "calibrated_hit_at_5": test_calibrated["all_outcome_hit_at_5"][outcome_name],
                "raw_hit_at_10": test_raw["all_outcome_hit_at_10"][outcome_name],
                "calibrated_hit_at_10": test_calibrated["all_outcome_hit_at_10"][outcome_name],
            })
        pd.DataFrame(outcome_rows).to_csv(
            args.output_dir / f"{data_dir.name}_all_outcome_hits.csv", index=False)
        reports.append(report)
        print(f"[{data_dir.name}] wrote {report_path}", flush=True)
        del model, dataset, collected
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary = {
        "protocol": (f"subject-ID-grouped {100.0 * (1.0 - args.test_fraction):g}% "
                     f"validation / {100.0 * args.test_fraction:g}% sealed test; "
                     "frozen transformer; validation-only output calibration; "
                     "subject IDs may be record-level proxies when patient IDs are unavailable; "
                     f"expanded ranking metrics use a uniform reservoir capped at {args.expanded_metric_limit:,} "
                     "test targets"),
        "checkpoint": str(args.checkpoint.resolve()),
        "sources": [{
            "source": report["source"],
            "split": report["split"],
            "selected_calibration": report["selected_calibration"],
            "test_split_final": report["test_split_final"],
        } for report in reports],
    }
    json_dump(summary, args.output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
