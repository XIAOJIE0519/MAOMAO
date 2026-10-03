#!/usr/bin/env python3
"""Exploratory paired classic-index vs frozen-MAOMAO benchmarks at end of surgery.

The available INSPIRE outcomes are recorded ICU transfer, vasopressor starts,
and threshold-based severe MAP hypotension. The report labels them as proxies
where appropriate and uses the existing patient-level MAOMAO validation split,
so this is a development analysis rather than a final independent test.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from maomao.data.event_sequence import EventSequenceDataset, collate_event_sequences  # noqa: E402
from scripts.diagnostics.evaluate_external_validation import build_model  # noqa: E402


def sas_score(min_map: float, min_hr: float, ebl: float) -> int:
    map_points = 3 if min_map >= 70 else 2 if min_map >= 55 else 1 if min_map >= 40 else 0
    hr_points = 3 if min_hr <= 55 else 2 if min_hr <= 70 else 1 if min_hr <= 110 else 0
    ebl_points = 3 if ebl <= 100 else 2 if ebl <= 600 else 1 if ebl <= 1000 else 0
    return map_points + hr_points + ebl_points


def collect_score_inputs(operations: pd.DataFrame, source_path: Path) -> pd.DataFrame:
    opids = set(operations.op_id.astype("int64"))
    ranges = operations.set_index("op_id")[["opstart_time", "opend_time", "orin_time"]].to_dict("index")
    accum: dict[int, dict[str, float]] = {}
    proc = subprocess.Popen(["gzip", "-cd", str(source_path)], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    try:
        for chunk in pd.read_csv(proc.stdout, chunksize=1_000_000):
            chunk = chunk[chunk.op_id.isin(opids) & chunk.item_name.isin(
                {"hr", "art_mbp", "nibp_mbp", "art_sbp", "nibp_sbp", "ebl"})]
            for row in chunk.itertuples(index=False):
                opid = int(row.op_id)
                bounds = ranges[opid]
                start = bounds["opstart_time"]
                if pd.isna(start):
                    start = bounds["orin_time"]
                end = bounds["opend_time"]
                if pd.isna(start) or pd.isna(end) or row.chart_time < start or row.chart_time > end:
                    continue
                try:
                    value = float(row.value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(value):
                    continue
                x = accum.setdefault(opid, {})
                if row.item_name == "hr" and 0 < value <= 250:
                    x["min_hr"] = min(x.get("min_hr", value), value)
                    if row.chart_time <= end and end - row.chart_time <= 15:
                        if row.chart_time >= x.get("latest_hr_time", -np.inf):
                            x["latest_hr"], x["latest_hr_time"] = value, float(row.chart_time)
                elif row.item_name in {"art_mbp", "nibp_mbp"} and 0 < value <= 250:
                    x["min_map"] = min(x.get("min_map", value), value)
                    if row.chart_time <= end and end - row.chart_time <= 15:
                        if row.chart_time >= x.get("latest_map_time", -np.inf):
                            x["latest_map"], x["latest_map_time"] = value, float(row.chart_time)
                elif row.item_name == "ebl" and 0 <= value <= 50000:
                    x["max_ebl"] = max(x.get("max_ebl", value), value)
                elif row.item_name in {"art_sbp", "nibp_sbp"} and 30 <= value <= 300:
                    if row.chart_time <= end and end - row.chart_time <= 15:
                        if row.chart_time >= x.get("latest_sbp_time", -np.inf):
                            x["latest_sbp"], x["latest_sbp_time"] = value, float(row.chart_time)
    finally:
        proc.stdout.close()
        status = proc.wait()
    if status not in (0, 2):
        raise RuntimeError(f"Could not read {source_path}; gzip exited {status}")
    rows = []
    for opid in operations.op_id.astype("int64"):
        x = accum.get(int(opid), {})
        rows.append({"op_id": int(opid), **x})
    return operations[["op_id"]].merge(pd.DataFrame(rows), on="op_id", how="left")


def bootstrap_metrics(y: np.ndarray, predictions: dict[str, np.ndarray],
                      subject: np.ndarray, repeats: int, seed: int) -> dict:
    unique = np.unique(subject)
    groups = {s: np.flatnonzero(subject == s) for s in unique}
    rng = np.random.default_rng(seed)
    point = {name: {"auroc": float(roc_auc_score(y, score)),
                    "auprc": float(average_precision_score(y, score))}
             for name, score in predictions.items()}
    samples = {name: {metric: [] for metric in ("auroc", "auprc")}
               for name in predictions}
    deltas = {name: {metric: [] for metric in ("auroc", "auprc")}
              for name in predictions if name != "MAOMAO (native 24h ICU-transfer head)"}
    maomao_name = next(name for name in predictions if name.startswith("MAOMAO (native "))
    for _ in range(repeats):
        draw = rng.choice(unique, size=len(unique), replace=True)
        row_ids = np.concatenate([groups[s] for s in draw])
        yy = y[row_ids]
        if yy.min() == yy.max():
            continue
        metrics = {}
        for name, score in predictions.items():
            metrics[name] = {
                "auroc": roc_auc_score(yy, score[row_ids]),
                "auprc": average_precision_score(yy, score[row_ids]),
            }
            for metric in ("auroc", "auprc"):
                samples[name][metric].append(metrics[name][metric])
        for name in deltas:
            for metric in ("auroc", "auprc"):
                deltas[name][metric].append(metrics[name][metric] - metrics[maomao_name][metric])
    output = {}
    for name in predictions:
        output[name] = {metric: {
            "estimate": point[name][metric],
            "ci95": [float(np.quantile(samples[name][metric], 0.025)),
                     float(np.quantile(samples[name][metric], 0.975))],
        } for metric in ("auroc", "auprc")}
    output["paired_difference_vs_MAOMAO"] = {
        name: {metric: {
            "estimate": point[name][metric] - point[maomao_name][metric],
            "ci95": [float(np.quantile(deltas[name][metric], 0.025)),
                     float(np.quantile(deltas[name][metric], 0.975))],
        } for metric in ("auroc", "auprc")} for name in deltas
    }
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path,
                    default=ROOT / "data/perioperative_event_sequences_v5_richctx_static7")
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "outputs/final_experiment_results_20260923/full_maomao_reference/best_model.pt")
    ap.add_argument("--split", type=Path, default=None,
                    help="Patient split NPZ for the existing internal validation; test datasets use all rows")
    ap.add_argument("--operations", type=Path, default=ROOT / "data/train_data/operations.csv.gz")
    ap.add_argument("--vitals", type=Path, default=ROOT / "data/train_data/vitals.csv.gz")
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "outputs/classic_score_comparison/sas_maomao_internal_validation")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--bootstrap-repeats", type=int, default=300)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["args"]
    dataset = EventSequenceDataset(args.data_dir, config.get("block_size", 256),
                                   config.get("window_stride", 128), dynamic_windows=False)
    admissions = pd.read_csv(args.data_dir / "admissions.csv")
    is_sealed_test = dataset.split == "test"
    if is_sealed_test:
        validation_episode_ids = np.arange(len(admissions), dtype=np.int64)
        validation_window_indices = np.arange(len(dataset), dtype=np.int64)
        split_description = "patient-disjoint sealed test; excluded from training and checkpoint selection"
    else:
        split_path = args.split or (ROOT / "outputs/final_experiment_results_20260923/full_maomao_reference/patient_validation_split.npz")
        split = np.load(split_path)
        val_patients = set(split["validation_patients"].astype(str))
        validation_episode_ids = np.flatnonzero(
            admissions.subject_id.astype(str).isin(val_patients).to_numpy())
        validation_window_indices = split["validation_window_indices"]
        split_description = "existing patient-level 90/10 development/validation split; validation influenced checkpoint selection"

    vocab = dataset.meta["token_vocabulary"]
    surgery_end_token = int(vocab["event:surgery_end"])
    task_specs = {
        "icu_transfer_24h": (2, dataset.meta["outcome_vocabulary"].index("icu_transfer")),
        "vasopressor_start_6h": (1, dataset.meta["outcome_vocabulary"].index("vasopressor_start")),
        "severe_map_hypotension_6h": (1, dataset.meta["outcome_vocabulary"].index("severe_map_hypotension")),
    }
    ptr = dataset.ptr
    token_ids = dataset.token_id
    times = dataset.time_min

    windows_by_episode: dict[int, list[int]] = {}
    for wi in validation_window_indices:
        windows_by_episode.setdefault(int(dataset.window_admission[wi]), []).append(int(wi))

    episode_position: dict[int, tuple[int, int]] = {}
    for ep in validation_episode_ids:
        lo, hi = int(ptr[ep]), int(ptr[ep + 1])
        local_tokens = np.asarray(token_ids[lo:hi])
        end_marks = np.flatnonzero(local_tokens == surgery_end_token)
        if not len(end_marks):
            continue
        event_position = int(end_marks[-1])
        event_time = float(times[lo + event_position])
        local_end = int(np.searchsorted(times[lo:hi], event_time, side="right") - 1)
        covering = [wi for wi in windows_by_episode.get(int(ep), [])
                    if int(dataset.window_start[wi]) <= local_end <
                    int(dataset.window_start[wi] + dataset.window_length[wi])]
        if covering:
            window = min(covering, key=lambda wi: int(dataset.window_start[wi]))
            episode_position[int(ep)] = (window, local_end - int(dataset.window_start[window]))

    selected_windows = sorted({window for window, _ in episode_position.values()})
    by_window: dict[int, list[tuple[int, int]]] = {}
    for ep, (window, position) in episode_position.items():
        by_window.setdefault(window, []).append((ep, position))

    model = build_model(dataset, checkpoint, device)
    model.eval()
    maomao_probability: dict[str, dict[int, float]] = {name: {} for name in task_specs}
    label: dict[str, dict[int, int]] = {name: {} for name in task_specs}
    loader = DataLoader(Subset(dataset, selected_windows), batch_size=args.batch_size,
                        shuffle=False, num_workers=0, collate_fn=collate_event_sequences,
                        pin_memory=device.type == "cuda")
    with torch.inference_mode():
        for batch_no, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            output = model(batch)
            lo_window = batch_no * args.batch_size
            batch_windows = selected_windows[lo_window:lo_window + len(batch["token_id"])]
            for local_i, window in enumerate(batch_windows):
                for ep, position in by_window[window]:
                    for task_name, (horizon_index, outcome_index) in task_specs.items():
                        if not bool(batch["trajectory_mask"][local_i, position, horizon_index]):
                            continue
                        p = torch.sigmoid(output.trajectory_logits[
                            local_i, position, horizon_index, outcome_index])
                        maomao_probability[task_name][ep] = float(p.cpu())
                        label[task_name][ep] = int(
                            batch["trajectory_target"][local_i, position,
                                                       horizon_index, outcome_index] > 0)

    val_rows = admissions.iloc[validation_episode_ids].copy().reset_index(drop=True)
    score_rows = collect_score_inputs(val_rows, args.vitals)
    val_rows = val_rows.merge(score_rows, on="op_id", how="left", validate="one_to_one")
    val_rows["sas"] = [sas_score(r.min_map, r.min_hr, r.max_ebl)
                        if pd.notna(r.min_map) and pd.notna(r.min_hr) and pd.notna(r.max_ebl)
                        else np.nan for r in val_rows.itertuples(index=False)]
    val_rows["sequence_id"] = validation_episode_ids
    for task_name in task_specs:
        val_rows[f"maomao_{task_name}_probability"] = val_rows.sequence_id.map(
            maomao_probability[task_name])
        val_rows[task_name] = val_rows.sequence_id.map(label[task_name])
    val_rows["score_complete"] = val_rows.sas.notna()
    val_rows["shock_index"] = val_rows.latest_hr / val_rows.latest_sbp
    val_rows["modified_shock_index"] = val_rows.latest_hr / val_rows.latest_map
    groups = {}
    comparator_specs = [
        ("SAS", "score_complete", "sas", -1.0, "icu_transfer_24h"),
        ("Shock Index", "shock_index", "shock_index", 1.0, "vasopressor_start_6h"),
        ("Shock Index", "shock_index", "shock_index", 1.0, "severe_map_hypotension_6h"),
        ("Modified Shock Index", "modified_shock_index", "modified_shock_index", 1.0,
         "vasopressor_start_6h"),
        ("Modified Shock Index", "modified_shock_index", "modified_shock_index", 1.0,
         "severe_map_hypotension_6h"),
    ]
    for name, condition_col, score_col, direction, task_name in comparator_specs:
        condition = val_rows[condition_col].notna() if condition_col != "score_complete" else val_rows.score_complete
        maomao_col = f"maomao_{task_name}_probability"
        matched = val_rows[condition & val_rows[maomao_col].notna() &
                           val_rows[task_name].notna()].copy()
        if matched[task_name].nunique() < 2:
            continue
        y = matched[task_name].to_numpy(dtype=np.int8)
        score_name = f"{name} (risk rank)"
        hname = f"MAOMAO (native {task_name.replace('_', ' ')} head)"
        prediction = {
            score_name: direction * matched[score_col].to_numpy(dtype=float),
            hname: matched[maomao_col].to_numpy(dtype=float),
        }
        group = {
            "matched_episodes": int(len(matched)),
            "matched_patients": int(matched.subject_id.nunique()),
            "events": int(y.sum()),
            "event_rate": float(y.mean()),
            "metrics_patient_cluster_paired_bootstrap": bootstrap_metrics(
                y, prediction, matched.subject_id.astype(str).to_numpy(),
                args.bootstrap_repeats, 4251),
        }
        groups.setdefault(task_name, {})[name] = group
    summary = {
        "status": "sealed_independent_test" if is_sealed_test else "exploratory_internal_validation",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "patient_split": split_description,
        "index": "Surgical Apgar Score calculated from intraoperative min MAP, min HR, and max recorded EBL",
        "landmark": "end of surgery; MAOMAO and score use information available by this point",
        "endpoint_limits": {
            "icu_transfer_24h": "ICU transfer may be planned; this is not the original SAS 30-day major-complication/death composite",
            "vasopressor_start_6h": "medication-derived event includes documented starts, not independently adjudicated shock",
            "severe_map_hypotension_6h": "threshold-based severe MAP hypotension signal, not an adjudicated complication",
        },
        "sas_cases_with_all_inputs": int(val_rows.score_complete.sum()),
        ("test_episodes" if is_sealed_test else "validation_episodes"): int(len(val_rows)),
        "latest_vital_window": "last recorded HR and BP values within 15 minutes before surgery end",
        "comparisons_by_endpoint": groups,
        "sas_point_distribution": {str(k): int(v) for k, v in
                                   val_rows.loc[val_rows.score_complete, "sas"].value_counts().sort_index().items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    val_rows.to_csv(args.output_dir / "validation_patient_episodes.csv", index=False)
    (args.output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
