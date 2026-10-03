#!/usr/bin/env python3
"""External evaluation for the V5 dual-timescale MAOMAO checkpoint.

This keeps the MAOMAO weights frozen, uses a patient-disjoint 90/10 split, and
reports raw versus calibration-only results. It deliberately decodes time from
event-specific hazards instead of reading the legacy shared log-normal head.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from maomao.data.event_sequence import EventSequenceDataset, collate_event_sequences  # noqa: E402
from maomao.evaluation.event_metrics import event_metric_report  # noqa: E402
from maomao.models.event_maomao import dual_timescale_expected_wait  # noqa: E402
from scripts.diagnostics.evaluate_external_validation import (  # noqa: E402
    build_model, check_contract, patient_split,
)


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    log_temperature = torch.zeros((), device=logits.device, requires_grad=True)
    optimizer = torch.optim.Adam([log_temperature], lr=0.05)
    for _ in range(120):
        calibrated = logits / log_temperature.exp().clamp(0.15, 6.0)
        positive = calibrated.masked_fill(~targets, -torch.inf)
        loss = (torch.logsumexp(calibrated, -1) - torch.logsumexp(positive, -1)).mean()
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    return float(log_temperature.detach().exp().clamp(0.15, 6.0))


@torch.no_grad()
def collect(model, dataset, indices, device, batch_size, max_rows, label):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_event_sequences,
                        pin_memory=device.type == "cuda")
    events, targets, waits, dts, observed = [], [], [], [], []
    rows = 0
    total_batches = (len(indices) + batch_size - 1) // batch_size
    for batch_no, batch in enumerate(loader, 1):
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        output = model(batch)
        event_valid = batch["loss_mask"].bool() & (batch["target_set"].sum(-1) > 0)
        time_valid = batch["time_mask"].bool()
        if event_valid.any():
            events.append(output.logits[event_valid].float().cpu())
            targets.append(batch["target_set"][event_valid].bool().cpu())
        if time_valid.any():
            if output.fine_hazard_logits is None:
                raise RuntimeError("Checkpoint does not contain dual-timescale hazard heads")
            waits.append(dual_timescale_expected_wait(
                output.fine_hazard_logits[time_valid],
                output.long_hazard_logits[time_valid],
                output.tail_mu[time_valid], output.tail_log_sigma[time_valid]).float().cpu())
            dts.append(batch["target_dt_hours"][time_valid].float().cpu())
            observed.append(event_valid[time_valid].cpu())
        rows += int(event_valid.sum())
        if batch_no == 1 or batch_no % 10 == 0 or batch_no == total_batches:
            print(f"[{label}] batch={batch_no}/{total_batches} valid_rows={rows}", flush=True)
        if rows >= max_rows:
            break
    return {
        "logits": torch.cat(events)[:max_rows], "targets": torch.cat(targets)[:max_rows],
        "wait_by_event": torch.cat(waits)[:max_rows], "dt": torch.cat(dts)[:max_rows],
        "observed": torch.cat(observed)[:max_rows],
    }


def score(collected: dict, names: list[str], temperature: float, time_scale: float,
           bootstrap_repeats: int) -> dict:
    print(f"[score] rows={len(collected['logits'])} temperature={temperature:.5f} "
          f"time_scale={time_scale:.5f} bootstrap={bootstrap_repeats}", flush=True)
    logits = collected["logits"] / float(temperature)
    report = event_metric_report(logits, collected["targets"], names,
                                 bootstrap_repeats=bootstrap_repeats)
    targets = collected["targets"].float()
    predicted_wait = (collected["wait_by_event"] * targets).sum(-1) / targets.sum(-1).clamp_min(1)
    observed = collected["observed"]
    if observed.any():
        error = predicted_wait[observed] * float(time_scale) - collected["dt"][observed]
        report.update({
            "time_targets": int(observed.sum()),
            "time_mae_hours": float(error.abs().mean()),
            "time_rmse_hours": float(error.square().mean().sqrt()),
            "time_scale": float(time_scale),
        })
    else:
        report.update({"time_targets": 0, "time_mae_hours": None,
                       "time_rmse_hours": None, "time_scale": float(time_scale)})
    print(f"[score] complete event_targets={report.get('event_targets')} "
          f"micro_auprc={report.get('micro_auprc')}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training_dir", type=Path, required=True)
    parser.add_argument("--data_dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_rows", type=int, default=200000)
    parser.add_argument("--bootstrap_repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if abs(args.test_fraction - 0.1) > 1e-9:
        raise ValueError("The unified external protocol requires a 90/10 calibration/test split")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_reports = []
    for data_dir in args.data_dirs:
        print(f"[source] loading {data_dir.name}", flush=True)
        dataset = EventSequenceDataset(data_dir, checkpoint["args"].get("block_size", 256),
                                       checkpoint["args"].get("window_stride", 128))
        contract = check_contract(data_dir, args.training_dir)
        validation, test, split = patient_split(dataset, data_dir, args.output_dir,
                                                 args.test_fraction, args.seed)
        model = build_model(dataset, checkpoint, device)
        print(f"[source] split calibration={len(validation)} test={len(test)}", flush=True)
        calibration = collect(model, dataset, validation, device, args.batch_size, args.max_rows,
                              f"{data_dir.name}:calibration")
        test_rows = collect(model, dataset, test, device, args.batch_size, args.max_rows,
                            f"{data_dir.name}:test")
        print(f"[{data_dir.name}] fitting temperature and time scale", flush=True)
        temperature = fit_temperature(calibration["logits"].to(device), calibration["targets"].to(device))
        cal_targets = calibration["targets"].float()
        cal_pred = (calibration["wait_by_event"] * cal_targets).sum(-1) / cal_targets.sum(-1).clamp_min(1)
        cal_observed = calibration["observed"]
        time_scale = float((calibration["dt"][cal_observed].clamp_min(1e-4).log() -
                            cal_pred[cal_observed].clamp_min(1e-4).log()).mean().exp()) if cal_observed.any() else 1.0
        names = dataset.meta["outcome_vocabulary"]
        raw = score(test_rows, names, 1.0, 1.0, args.bootstrap_repeats)
        calibrated = score(test_rows, names, temperature, time_scale, args.bootstrap_repeats)
        report = {"source": data_dir.name, "checkpoint": str(args.checkpoint.resolve()),
                  "contract": contract, "split": split,
                  "calibration": {"temperature": temperature, "time_scale": time_scale},
                  "test_raw": raw, "test_calibrated": calibrated}
        (args.output_dir / f"{data_dir.name}_dual_evaluation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2))
        all_reports.append(report)
        print(json.dumps({"source": data_dir.name, "raw": raw, "calibrated": calibrated},
                         ensure_ascii=False), flush=True)
    (args.output_dir / "dual_summary.json").write_text(
        json.dumps({"protocol": "patient-disjoint 90% calibration / 10% sealed test",
                    "reports": all_reports}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
