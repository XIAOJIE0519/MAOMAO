#!/usr/bin/env python3
"""Frozen INSPIRE-trained MAOMAO versus feasible scores on MOVER episodes.

MOVER's recorded OR exit is used as an end-of-case landmark proxy. Scores and
labels are restricted to the corresponding common episode timeline. Missing
static fields are explicitly adapted and sparse/unavailable outcomes are not
imputed from unrelated complication codes.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from maomao.data.event_sequence import EventSequenceDataset, collate_event_sequences  # noqa: E402
from scripts.diagnostics.evaluate_external_validation import build_model  # noqa: E402


def cdf_at(rec, idx: int, hours: float, trajectory_idx: dict[float, int]) -> float:
    if hours in trajectory_idx:
        return float(rec["trajectory"][trajectory_idx[hours], idx])
    fine, long = rec.get("fine"), rec.get("long")
    if fine is None or long is None:
        return np.nan
    surv = 1.0
    if hours <= 2:
        n = int(round(hours * 12))
        return 1.0 - float(np.prod(1.0 - fine[idx, :n]))
    surv *= float(np.prod(1.0 - fine[idx]))
    if hours <= 24:
        n = int(round((hours - 2.0) * 2))
        return 1.0 - surv * float(np.prod(1.0 - long[idx, :n]))
    surv *= float(np.prod(1.0 - long[idx]))
    sigma = max(float(np.exp(rec["tail_log_sigma"][idx])), 1e-6)
    z = (math.log(max(hours - 24.0, 1e-6)) - float(rec["tail_mu"][idx])) / sigma
    return 1.0 - surv * (1.0 - float(ndtr(z)))


def cluster_bootstrap(y, preds, patient, repeats=300, seed=59331):
    unique = np.unique(patient)
    groups = {s: np.flatnonzero(patient == s) for s in unique}
    rng = np.random.default_rng(seed)
    names = list(preds)
    point = {n: (float(roc_auc_score(y, preds[n])), float(average_precision_score(y, preds[n]))) for n in names}
    samples = {n: [[], []] for n in names}
    diffs = {n: [[], []] for n in names if n != "MAOMAO"}
    for _ in range(repeats):
        draw = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([groups[s] for s in draw])
        yy = y[ix]
        if yy.min() == yy.max():
            continue
        for n in names:
            auc = roc_auc_score(yy, preds[n][ix]); ap = average_precision_score(yy, preds[n][ix])
            samples[n][0].append(auc); samples[n][1].append(ap)
        for n in diffs:
            for j in range(2):
                diffs[n][j].append(samples[n][j][-1] - samples["MAOMAO"][j][-1])
    result = {}
    for n in names:
        result[n] = {
            "auroc": {"estimate": point[n][0], "ci95": np.quantile(samples[n][0], [0.025, 0.975]).tolist()},
            "auprc": {"estimate": point[n][1], "ci95": np.quantile(samples[n][1], [0.025, 0.975]).tolist()},
        }
    result["score_minus_maomao"] = {
        n: {
            "auroc": {"estimate": point[n][0] - point["MAOMAO"][0], "ci95": np.quantile(diffs[n][0], [0.025, 0.975]).tolist()},
            "auprc": {"estimate": point[n][1] - point["MAOMAO"][1], "ci95": np.quantile(diffs[n][1], [0.025, 0.975]).tolist()},
        } for n in diffs
    }
    return result


def prepare_dataset(data_dir: Path, train_dir: Path, checkpoint: dict):
    train_meta = json.loads((train_dir / "event_sequence_meta.json").read_text())
    dataset = EventSequenceDataset(
        data_dir, checkpoint["args"].get("block_size", 256),
        checkpoint["args"].get("window_stride", 128), dynamic_windows=False,
        outcome_names=train_meta["outcome_vocabulary"])
    train_vocab = train_meta["token_vocabulary"]
    mover_vocab = dataset.meta["token_vocabulary"]
    missing_tokens = [k for k in mover_vocab if k not in train_vocab]
    if missing_tokens:
        raise ValueError(f"MOVER contains tokens outside the frozen training vocabulary: {missing_tokens[:10]}")
    # MOVER omits 36 training-only baseline/summary tokens, so its serialized
    # IDs shift from the first event token onward. Remap by token name before
    # applying the frozen checkpoint; never feed the external integer IDs as-is.
    external_names = {int(v): k for k, v in mover_vocab.items()}
    remap = np.empty(max(external_names) + 1, dtype=np.int32)
    for external_id, name in external_names.items():
        remap[external_id] = int(train_vocab[name])
    dataset.token_id = remap[np.asarray(dataset.token_id)].astype(np.int32, copy=False)
    dataset.meta["token_vocabulary"] = train_vocab
    dataset.num_tokens = max(train_vocab.values()) + 1
    dataset.num_static = len(train_meta["static_features"])
    dataset.outcome_family_ids = np.asarray(train_meta["outcome_to_family"], dtype=np.int16)
    dataset.num_event_families = int(dataset.outcome_family_ids.max()) + 1
    dataset.meta["outcome_to_family"] = train_meta["outcome_to_family"]
    dataset.meta["num_static"] = dataset.num_static
    admissions = pd.read_csv(data_dir / "admissions.csv", dtype={"subject_id": "string"})
    age = pd.to_numeric(admissions.age, errors="coerce").to_numpy(dtype=np.float32)
    male = pd.to_numeric(admissions.male, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    weight = pd.to_numeric(admissions.weight_kg, errors="coerce").to_numpy(dtype=np.float32)
    height = pd.to_numeric(admissions.height_cm, errors="coerce").to_numpy(dtype=np.float32)
    asa = pd.to_numeric(admissions.asa, errors="coerce").to_numpy(dtype=np.float32)
    bmi = weight / np.square(height / 100.0)
    static = np.column_stack((age / 100.0, male, bmi / 40.0, asa / 6.0,
                              np.zeros(len(admissions), dtype=np.float32),
                              weight / 150.0, height / 200.0)).astype(np.float32)
    static[~np.isfinite(static)] = 0.0
    dataset.static = static
    return dataset, admissions, train_meta


def infer_or_exit(dataset, admissions, checkpoint, device, batch_size):
    token_id = dataset.meta["token_vocabulary"].get("event:or_exit")
    if token_id is None:
        raise ValueError("MOVER sequence has no OR-exit token")
    windows: dict[int, list[int]] = defaultdict(list)
    for wi, ep in enumerate(dataset.window_admission):
        windows[int(ep)].append(int(wi))
    positions = {}
    for ep in range(len(admissions)):
        lo, hi = int(dataset.ptr[ep]), int(dataset.ptr[ep + 1])
        ids = np.asarray(dataset.token_id[lo:hi])
        marks = np.flatnonzero(ids == token_id)
        if not len(marks):
            continue
        landmark = float(dataset.time_min[lo + int(marks[-1])])
        pos = int(np.searchsorted(dataset.time_min[lo:hi], landmark, side="right") - 1)
        covering = [wi for wi in windows[ep] if dataset.window_start[wi] <= pos <
                    dataset.window_start[wi] + dataset.window_length[wi]]
        if covering:
            wi = min(covering, key=lambda x: dataset.window_start[x])
            positions[ep] = (wi, pos - int(dataset.window_start[wi]), landmark)
    selected = sorted({v[0] for v in positions.values()})
    by_window = defaultdict(list)
    for ep, v in positions.items():
        by_window[v[0]].append((ep, v[1], v[2]))
    model = build_model(dataset, checkpoint, device)
    model.eval()
    outputs = {}
    loader = DataLoader(Subset(dataset, selected), batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_event_sequences,
                        pin_memory=device.type == "cuda")
    with torch.inference_mode():
        for bno, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(batch)
            ws = selected[bno * batch_size:bno * batch_size + len(batch["token_id"])]
            for bi, wi in enumerate(ws):
                for ep, pos, landmark in by_window[wi]:
                    rec = {"landmark": landmark,
                           "trajectory": torch.sigmoid(out.trajectory_logits[bi, pos]).float().cpu().numpy()}
                    if out.fine_hazard_logits is not None:
                        rec["fine"] = torch.sigmoid(out.fine_hazard_logits[bi, pos].float()).cpu().numpy()
                        rec["long"] = torch.sigmoid(out.long_hazard_logits[bi, pos].float()).cpu().numpy()
                        rec["tail_mu"] = out.tail_mu[bi, pos].float().cpu().numpy()
                        rec["tail_log_sigma"] = out.tail_log_sigma[bi, pos].float().cpu().numpy()
                    outputs[ep] = rec
    return outputs


def query_mover_features(db_path: Path, dataset, admissions, landmark_by_ep):
    """Retrieve the last HR/SBP/MAP and pre-OR creatinine observations."""
    con = sqlite3.connect(str(db_path))
    con.execute("CREATE TEMP TABLE lm(ep INTEGER PRIMARY KEY, lm_bin INTEGER, pre_bin INTEGER)")
    rows = []
    for ep, lm in landmark_by_ep.items():
        pre = pd.to_numeric(pd.Series([admissions.iloc[ep].or_in_min]), errors="coerce").iloc[0]
        rows.append((int(ep), int(math.floor(lm / 5)),
                     int(math.floor(float(pre) / 5)) if np.isfinite(pre) else -1))
    con.executemany("INSERT INTO lm VALUES(?,?,?)", rows)
    features = ["ward_vitals:hr", "vitals:nibp_sbp", "vitals:nibp_mbp",
                "labs:creatinine", "labs:hb"]
    query = """
      SELECT o.ep,o.feature,o.bin,o.value_sum*1.0/o.value_count
      FROM obs o JOIN lm l ON l.ep=o.ep
      WHERE ((o.feature IN (?,?,?) AND o.bin BETWEEN l.lm_bin-3 AND l.lm_bin)
          OR (o.feature IN (?,?) AND o.bin BETWEEN l.pre_bin-288 AND l.pre_bin))
      ORDER BY o.ep,o.feature,o.bin
    """
    feature_rows = defaultdict(dict)
    for ep, feature, bin_no, value in con.execute(query, (*features[:3], *features[3:])):
        feature_rows[int(ep)][feature] = (int(bin_no), float(value))
    con.close()
    return feature_rows


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    dataset, admissions, train_meta = prepare_dataset(args.data_dir, args.train_dir, checkpoint)
    predictions = infer_or_exit(dataset, admissions, checkpoint, device, args.batch_size)
    feature_rows = query_mover_features(args.staging_db, dataset, admissions,
                                        {ep: rec["landmark"] for ep, rec in predictions.items()})
    vocab = dataset.meta["outcome_vocabulary"]
    endpoints = [
        ("hypotension_15m", 0.25, ["severe_map_hypotension"]),
        ("hypotension_30m", 0.5, ["severe_map_hypotension"]),
        ("hypotension_60m", 1.0, ["severe_map_hypotension"]),
        ("hypotension_6h", 6.0, ["severe_map_hypotension"]),
        ("pressor_30m", 0.5, ["vasopressor_start"]),
        ("pressor_1h", 1.0, ["vasopressor_start"]),
        ("pressor_6h", 6.0, ["vasopressor_start"]),
        ("aki23_48h", 48.0, ["aki_stage_2_signal", "aki_stage_3_signal"]),
        ("aki23_7d", 168.0, ["aki_stage_2_signal", "aki_stage_3_signal"]),
        ("troponin_24h", 24.0, ["troponin_i_elevation"]),
        ("death_inhospital_30d", 720.0, ["inhospital_death"]),
    ]
    specs_by_name = {x[0]: x for x in endpoints}
    trajectory_idx = {float(h): i for i, h in enumerate(dataset.trajectory_horizons_hours)}
    outv = dataset.meta["outcome_vocabulary"]
    labels = {}
    rows = []
    for ep, rec in predictions.items():
        lo, hi = int(dataset.ptr[ep]), int(dataset.ptr[ep + 1])
        times = np.asarray(dataset.time_min[lo:hi])
        raw = np.asarray(dataset.outcome[lo:hi])
        mapped = np.full(raw.shape, -1, dtype=np.int16)
        valid = (raw >= 0) & (raw < len(dataset.outcome_remap))
        mapped[valid] = dataset.outcome_remap[raw[valid]]
        landmark = rec["landmark"]
        follow = float(times[-1] - landmark)
        adm = admissions.iloc[ep]
        f = feature_rows.get(ep, {})
        def value(key):
            item = f.get(key)
            return item[1] if item is not None else np.nan
        hr, sbp, mbp = value("ward_vitals:hr"), value("vitals:nibp_sbp"), value("vitals:nibp_mbp")
        row = {"ep": ep, "subject_id": str(adm.subject_id), "asa": pd.to_numeric(adm.asa, errors="coerce"),
               "age": pd.to_numeric(adm.age, errors="coerce"), "male": pd.to_numeric(adm.male, errors="coerce"),
               "department": str(adm.department), "current_hr": hr, "current_sbp": sbp, "current_map": mbp,
               "shock_index": hr / sbp if np.isfinite(hr) and np.isfinite(sbp) and sbp > 0 else np.nan,
               "modified_shock_index": hr / mbp if np.isfinite(hr) and np.isfinite(mbp) and mbp > 0 else np.nan,
               "preop_creatinine": value("labs:creatinine"), "preop_hb": value("labs:hb"),
               "followup_hours": follow}
        # MOVER history/urgency data are absent from this frozen sequence artifact;
        # keep a transparent four-factor available-component GS-AKI subtotal.
        dept = str(adm.department).lower()
        intraperitoneal_proxy = any(x in dept for x in ["general", "gastro", "colorectal", "hepat", "abdominal"])
        row["gs_aki_available4"] = (
            int(float(row["age"]) >= 56 if np.isfinite(row["age"]) else False) +
            int(float(row["male"]) > 0 if np.isfinite(row["male"]) else False) + int(intraperitoneal_proxy) +
            int(float(row["preop_creatinine"]) >= 1.2 if np.isfinite(row["preop_creatinine"]) else False))
        row["asa_plus_gs_aki_available4"] = (float(row["asa"]) - 1 + row["gs_aki_available4"]
                                             if np.isfinite(row["asa"]) else np.nan)
        for name, hours, outcomes in endpoints:
            indices = [outv.index(x) for x in outcomes if x in outv]
            if not indices:
                row["label_" + name] = np.nan; row["maomao_" + name] = np.nan
                continue
            event = (times > landmark) & np.isin(mapped, indices)
            horizon_min = hours * 60
            positive = bool(np.any(event & (times <= landmark + horizon_min)))
            complete = follow >= horizon_min
            if name == "death_inhospital_30d":
                # Survival to discharge is known for this in-hospital endpoint.
                death_min = pd.to_numeric(adm.death_min, errors="coerce")
                if not np.isfinite(death_min) or death_min > landmark + horizon_min:
                    complete = True
                if np.isfinite(death_min) and death_min <= landmark:
                    positive = False; complete = False
            row["label_" + name] = 1 if positive else (0 if complete else np.nan)
            risk = [cdf_at(rec, idx, hours, trajectory_idx) for idx in indices]
            row["maomao_" + name] = float(np.nanmax(risk))
        rows.append(row)
    scores = {
        "ASA-PS": ("asa", 1), "GS-AKI available 4/9 components": ("gs_aki_available4", 1),
        "ASA + GS-AKI available 4/9": ("asa_plus_gs_aki_available4", 1),
        "Shock Index": ("shock_index", 1), "Modified Shock Index": ("modified_shock_index", 1),
        "Current MAP": ("current_map", -1),
    }
    endpoints_scores = {
        **{k: ["Shock Index", "Modified Shock Index", "Current MAP"] for k, _, _ in endpoints[:7]},
        "aki23_48h": ["ASA-PS", "GS-AKI available 4/9 components", "ASA + GS-AKI available 4/9"],
        "aki23_7d": ["ASA-PS", "GS-AKI available 4/9 components", "ASA + GS-AKI available 4/9"],
        "troponin_24h": ["ASA-PS", "Current MAP"],
        "death_inhospital_30d": ["ASA-PS"],
    }
    df = pd.DataFrame(rows)
    report = {"status": "frozen_checkpoint_external_MOVER", "checkpoint": str(args.checkpoint.resolve()),
              "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
              "episodes": int(len(admissions)), "patients": int(admissions.subject_id.nunique()),
              "landmark": "event:or_exit; MOVER has no reliable surgery_end timestamp, so this is an OR-exit proxy",
              "adaptations": {
                  "static_model_inputs": "MAOMAO static vector rebuilt in training order; age, sex, ASA, weight, and height from MOVER; BMI derived; emergency unavailable and filled as 0",
                  "token_contract": "MOVER's 2,154 token names remapped into the frozen 2,190-token INSPIRE vocabulary; 36 training-only baseline/summary tokens are absent and never appear in MOVER episodes",
                  "gs_aki": "Only age, male, department-based intraperitoneal proxy, and pre-OR creatinine are available in this artifact (4/9); this subtotal is exploratory and not the validated GS-AKI score",
                  "outcomes": "AKI stage 2/3, pressure/pressor and troponin are threshold/recorded-event signals; ICU_ADMIN_FLAG is mapped to OR exit and cannot be treated as a prospective ICU outcome",
                  "unavailable": "This frozen MOVER sequence has no auditable RBC-transfusion, CRRT-start, or ventilation-start event labels; no paired comparisons are reported for these endpoints",
                  "death": "In-hospital death within 30 days after OR exit; live discharge before 30 days is a known negative; not complete all-cause 30-day follow-up",
              },
              "comparisons_by_endpoint": {}}
    for endpoint, names in endpoints_scores.items():
        ycol, hcol = "label_" + endpoint, "maomao_" + endpoint
        comps = []
        for name in names:
            col, direction = scores[name]
            part = df.loc[df[ycol].notna() & df[hcol].notna() & df[col].notna()].copy()
            if len(part) < 30 or part[ycol].nunique() < 2:
                continue
            y = part[ycol].to_numpy(dtype=np.int8)
            preds = {name: direction * pd.to_numeric(part[col]).to_numpy(dtype=float),
                     "MAOMAO": pd.to_numeric(part[hcol]).to_numpy(dtype=float)}
            comps.append({"comparator": name, "n_episodes": len(part), "n_patients": int(part.subject_id.nunique()),
                          "events": int(y.sum()), "event_rate": float(y.mean()),
                          "metrics_patient_cluster_bootstrap": cluster_bootstrap(
                              y, preds, part.subject_id.astype(str).to_numpy(),
                              args.bootstrap_repeats, args.bootstrap_seed + len(report["comparisons_by_endpoint"]) * 41 + len(comps))})
        report["comparisons_by_endpoint"][endpoint] = comps
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "mover_episode_predictions.csv", index=False)
    (args.output_dir / "mover_comparison_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/val_mover_v5")
    ap.add_argument("--train-dir", type=Path, default=ROOT / "data/perioperative_event_sequences_v5_richctx_static7")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/classic_score_comparison/independent_15pct_test/maomao_fresh/best_model.pt")
    ap.add_argument("--staging-db", type=Path, default=ROOT / "data/val_mover_staging.sqlite")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs/classic_score_comparison/independent_15pct_test/mover_results")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--bootstrap-repeats", type=int, default=300)
    ap.add_argument("--bootstrap-seed", type=int, default=59331)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
