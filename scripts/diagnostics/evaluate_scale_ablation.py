"""Full-coordinate scoring for size/output/context MAOMAO experiments."""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from maomao.data.event_sequence import EventSequenceDataset, collate_event_sequences
from maomao.data.scale_ablation import ProjectedOutcomeDataset
from maomao.models.event_maomao import dual_timescale_expected_wait
from maomao.evaluation.event_metrics import event_metric_report_with_subsample_ci
from scripts.diagnostics.evaluate_dual_timescale_internal import boot_ci
from scripts.diagnostics.scale_ablation_scope import *


def cache_model(name):
    manifest = read(OUT / "manifest.json")
    checkpoint = REFERENCE / "best_model.pt" if name == "reference" else OUT / "runs" / name / "best_model.pt"
    cache = OUT / "cache" / name
    cache.mkdir(parents=True, exist_ok=True)
    if (cache / "complete.json").exists():
        state = read(cache / "complete.json")
        if state["checkpoint_sha256"] == sha256(checkpoint) and state["manifest_sha256"] == sha256(OUT / "manifest.json"):
            for path in (cache / "logits.npy", cache / "wait_by_event.npy"):
                if not path.is_file() or np.load(path, mmap_mode="r").shape != tuple(state["shape"]):
                    raise RuntimeError("Cached full evaluation is incomplete")
            if np.load(cache / "target_dt_hours.npy", mmap_mode="r").shape != (state["shape"][0],):
                raise RuntimeError("Cached target time coordinates are incomplete")
            if sha256(cache / "target_dt_hours.npy") != state.get("target_time_sha256"):
                raise RuntimeError("Cached target time coordinates changed after scoring")
            return state
    device = torch.device("cuda")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["args"]
    dataset = EventSequenceDataset(DATA, 256, 128, dynamic_windows=False)
    indices = list(range(210))
    if cfg.get("outcome_projection_file"):
        indices = read(cfg["outcome_projection_file"])["indices"]
        dataset = ProjectedOutcomeDataset(dataset, cfg["outcome_projection_file"])
    split = np.load(REFERENCE / "patient_validation_split.npz")
    val_windows = split["validation_window_indices"]
    if name != "reference":
        actual_split = np.load(checkpoint.parent / "patient_validation_split.npz")
        for key in split.files:
            if not np.array_equal(split[key], actual_split[key]):
                raise RuntimeError(f"{name}: patient/window split differs on {key}")
    row_windows = np.load(ROWS / "validation_window_indices.npy", mmap_mode="r")
    row_positions = np.load(ROWS / "validation_positions.npy", mmap_mode="r")
    targets = np.load(ROWS / "validation_y.npy", mmap_mode="r")
    by_window = {}
    for row, window in enumerate(row_windows):
        by_window.setdefault(int(window), []).append(row)
    shape = (len(targets), dataset.num_outcomes)
    logits = np.lib.format.open_memmap(cache / "logits.npy", mode="w+", dtype="float32", shape=shape)
    wait = np.lib.format.open_memmap(cache / "wait_by_event.npy", mode="w+", dtype="float32", shape=shape)
    target_time = np.lib.format.open_memmap(cache / "target_dt_hours.npy", mode="w+", dtype="float32", shape=(len(targets),))
    seen = np.zeros(len(targets), dtype=bool)
    model = model_for(dataset, payload, device)
    parameter_count = sum(p.numel() for p in model.parameters())
    del payload
    loader = DataLoader(Subset(dataset, val_windows), batch_size=64, shuffle=False,
                        num_workers=0, collate_fn=collate_event_sequences, pin_memory=True)
    with torch.inference_mode():
        for batch_no, batch in enumerate(loader):
            rows, local = [], []
            for b, window in enumerate(val_windows[batch_no*64:(batch_no+1)*64]):
                selected = by_window.get(int(window), [])
                rows.extend(selected)
                local.extend([b] * len(selected))
            rows = np.asarray(rows, dtype=np.int64)
            positions = np.asarray(row_positions[rows], dtype=np.int64)
            local_tensor = torch.tensor(local, device=device)
            position_tensor = torch.tensor(positions, device=device)
            expected = np.array(targets[rows][:, indices], dtype=np.uint8, copy=True)
            actual = batch["target_set"][local, positions].numpy().astype(np.uint8)
            if not np.array_equal(actual, expected):
                raise RuntimeError("Projected labels differ at original full-validation coordinates")
            times = batch["target_dt_hours"][local, positions].numpy()
            reference_times = OUT / "cache/reference/target_dt_hours.npy"
            if name != "reference" and reference_times.exists():
                if not np.array_equal(np.load(reference_times, mmap_mode="r")[rows], times):
                    raise RuntimeError("An experiment changed the original next-event times")
            target_time[rows] = times
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(batch)
            logits[rows] = result.logits[local_tensor, position_tensor].float().cpu().numpy()
            predicted = dual_timescale_expected_wait(
                result.fine_hazard_logits[local_tensor, position_tensor],
                result.long_hazard_logits[local_tensor, position_tensor],
                result.tail_mu[local_tensor, position_tensor], result.tail_log_sigma[local_tensor, position_tensor])
            wait[rows] = predicted.cpu().numpy()
            seen[rows] = True
            if batch_no % 25 == 0 or batch_no + 1 == len(loader):
                print(f"{name}: batches={batch_no+1}/{len(loader)} original_rows_scored={int(seen.sum())}/{len(seen)}", flush=True)
            del result, predicted, batch
    if not seen.all():
        raise RuntimeError("Some original validation coordinates were never evaluated")
    logits.flush(); wait.flush(); target_time.flush()
    state = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
             "checkpoint_epoch": read_epoch(checkpoint),
             "parameter_count": parameter_count, "indices": indices, "shape": list(shape),
             "manifest_sha256": sha256(OUT / "manifest.json"), "all_original_rows_scored": True,
             "target_projection_verified": True,
             "target_time_sha256": sha256(cache / "target_dt_hours.npy")}
    write(cache / "complete.json", state)
    del model, logits, wait, target_time
    torch.cuda.empty_cache(); gc.collect()
    return state


def read_epoch(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    epoch = int(payload["epoch"])
    del payload
    return epoch


def score_cached(name, state, indices=None, report_name=None):
    report_name = report_name or name
    path = OUT / "metrics" / f"{report_name}.json"
    indices = indices or state["indices"]
    if path.exists():
        saved = read(path)
        if (saved.get("checkpoint_sha256") == state["checkpoint_sha256"] and
                saved.get("manifest_sha256") == sha256(OUT / "manifest.json") and
                saved.get("output_indices") == indices and saved.get("status") == "completed" and
                saved.get("train_output_classes_actual") == len(state["indices"]) and
                all(metric + "_95ci" in saved for metric in METRICS)):
            return
    targets = np.load(ROWS / "validation_y.npy", mmap_mode="r")
    projected = np.array(targets[:, indices], dtype=np.uint8, copy=True)
    eligible = projected.any(1)
    cache_columns = [state["indices"].index(i) for i in indices]
    logits = np.array(np.load(OUT / "cache" / name / "logits.npy", mmap_mode="r")[eligible][:, cache_columns], copy=True)
    selected_targets = projected[eligible]
    names = read(DATA / "event_sequence_meta.json")["outcome_vocabulary"]
    report = event_metric_report_with_subsample_ci(
        torch.from_numpy(logits), torch.from_numpy(selected_targets), [names[i] for i in indices],
        bootstrap_repeats=200, bootstrap_seed=4200, max_ci_rows=30_000)
    specification = read(OUT / f"specifications/vocab_{len(indices)}.json") if len(indices) != 210 else None
    manifest = read(OUT / "manifest.json")
    report.update(status="completed", model=report_name, data_dir=str(DATA),
                  checkpoint=state["checkpoint"], checkpoint_sha256=state["checkpoint_sha256"],
                  checkpoint_epoch=state["checkpoint_epoch"], parameter_count=state["parameter_count"],
                  train_rows_original=manifest["train_rows_full"], validation_rows_original=len(targets),
                  train_event_rows_eligible=(manifest["train_rows_full"] if name == "reference" or not specification
                                             else specification["eligible_rows"]["train"]),
                  train_rows_positive_in_evaluation_subset=(specification["eligible_rows"]["train"] if specification
                                                            else manifest["train_rows_full"]),
                  train_output_classes_actual=len(state["indices"]),
                  evaluation_rows=int(eligible.sum()), output_classes=len(indices), output_indices=indices,
                  all_original_rows_scored=True, evaluation_rows_full_patient_validation=True,
                  target_projection_verified=True, manifest_sha256=sha256(OUT / "manifest.json"),
                  target_time_sha256=state["target_time_sha256"],
                  ci_caveat="full eligible-row point estimates; 200 bootstrap replicates on 30k uniform rows with width scaling, approximate 95% CIs")
    write(path, report)
    print(f"{report_name}: full point metrics and 95% CI saved ({report['event_targets']} rows)", flush=True)
    del logits, projected, selected_targets
    gc.collect()


def time_report(name, state, indices=None, report_name=None):
    report_name = report_name or name
    indices = indices or state["indices"]
    path = OUT / "metrics" / f"{report_name}_time_scales.json"
    if path.exists():
        saved = read(path)
        if (saved.get("checkpoint_sha256") == state["checkpoint_sha256"] and
                saved.get("manifest_sha256") == sha256(OUT / "manifest.json") and
                saved.get("output_indices") == indices and
                saved.get("target_time_sha256") == state["target_time_sha256"] and
                saved.get("bootstrap_repeats") == 200 and
                saved.get("pooled_time_mae_reported") is False):
            return
    row_time = np.load(OUT / "cache" / name / "target_dt_hours.npy", mmap_mode="r")
    target = np.array(np.load(ROWS / "validation_y.npy", mmap_mode="r")[:, indices], dtype=np.float32, copy=True)
    columns = [state["indices"].index(i) for i in indices]
    predicted = np.load(OUT / "cache" / name / "wait_by_event.npy", mmap_mode="r")[:, columns]
    average = (predicted * target).sum(1) / np.maximum(1, target.sum(1))
    errors = np.abs(average - row_time)
    valid = target.any(1)
    groups = (("fine_0_to_2h", "<2 h", valid & (row_time < 2)),
              ("long_2h_to_tail_start", "2 to <24 h", valid & (row_time >= 2) & (row_time < 24)),
              ("tail_from_tail_start", "≥24 h", valid & (row_time >= 24)))
    report = {"model": report_name, "checkpoint_sha256": state["checkpoint_sha256"], "pooled_time_mae_reported": False,
              "event_target_rows": int(valid.sum()), "time_mae_by_scale": {},
              "output_indices": indices, "manifest_sha256": sha256(OUT / "manifest.json"),
              "target_time_sha256": state["target_time_sha256"], "bootstrap_repeats": 200,
              "ci_method": "200 percentile bootstrap resamples of all eligible event-target rows within each scale"}
    for i, (key, label, mask) in enumerate(groups):
        subset = errors[mask]
        report["time_mae_by_scale"][key] = {"label": label, "event_target_rows": len(subset),
                                          "mae_hours": float(subset.mean()) if len(subset) else None,
                                          "mae_hours_95ci": boot_ci(subset, seed=4200+i)}
    write(path, report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=("reference", *NAMES), required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    state = cache_model(args.name)
    score_cached(args.name, state)
    if args.name == "reference":
        for size in (50, 100, 150):
            score_cached("reference", state, read(OUT / f"specifications/vocab_{size}.json")["indices"], f"reference_vocab_{size}")
            time_report("reference", state, read(OUT / f"specifications/vocab_{size}.json")["indices"], f"reference_vocab_{size}")
    time_report(args.name, state)


if __name__ == "__main__":
    main()
