#!/usr/bin/env python3
"""Expanded frozen-MAOMAO versus bedside-score benchmarks on INSPIRE's sealed test.

This is a score-alignment analysis, not a clinical utility study. It keeps the
patient-disjoint test split fixed, requires complete outcome follow-up for
fixed-horizon negatives, and reports adapted score definitions explicitly.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import subprocess
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


def read_gzip_chunks(path: Path, usecols: list[str], chunksize: int = 1_000_000):
    proc = subprocess.Popen(["gzip", "-cd", str(path)], stdout=subprocess.PIPE)
    assert proc.stdout is not None
    try:
        yield from pd.read_csv(proc.stdout, usecols=usecols, chunksize=chunksize,
                               low_memory=False)
    finally:
        proc.stdout.close()
        rc = proc.wait()
        if rc not in (0, 2):
            raise RuntimeError(f"gzip failed for {path}: {rc}")


def _num(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def collect_or_inputs(ops: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Read intraoperative values once and derive Apgar, shock and MAP burden."""
    opids = set(pd.to_numeric(ops.op_id, errors="coerce").dropna().astype("int64"))
    bounds = {}
    for r in ops.itertuples(index=False):
        start = _num(getattr(r, "opstart_time", np.nan))
        if not np.isfinite(start):
            start = _num(getattr(r, "orin_time", np.nan))
        end = _num(getattr(r, "opend_time", np.nan))
        if not np.isfinite(end):
            end = _num(getattr(r, "orout_time", np.nan))
        bounds[int(r.op_id)] = (start, end)
    acc: dict[int, dict] = {}
    use_items = {"hr", "art_mbp", "nibp_mbp", "art_sbp", "nibp_sbp", "ebl"}
    for chunk in read_gzip_chunks(path, ["op_id", "chart_time", "item_name", "value"]):
        chunk["op_id"] = pd.to_numeric(chunk.op_id, errors="coerce")
        chunk = chunk[chunk.op_id.isin(opids) & chunk.item_name.isin(use_items)]
        for r in chunk.itertuples(index=False):
            opid = int(r.op_id)
            start, end = bounds[opid]
            t, v = _num(r.chart_time), _num(r.value)
            if not np.isfinite(t) or not np.isfinite(v) or not np.isfinite(start) or not np.isfinite(end):
                continue
            if t < start or t > end:
                continue
            x = acc.setdefault(opid, {"map_n": 0, "map_below65_n": 0,
                                      "map_deficit_sum": 0.0})
            item = r.item_name
            if item == "hr" and 0 < v <= 250:
                x["min_hr"] = min(x.get("min_hr", v), v)
                field = "latest_hr"
            elif item in {"art_mbp", "nibp_mbp"} and 0 < v <= 250:
                x["min_map"] = min(x.get("min_map", v), v)
                x["map_n"] += 1
                x["map_below65_n"] += int(v < 65)
                x["map_deficit_sum"] += max(65 - v, 0)
                field = "latest_map"
            elif item in {"art_sbp", "nibp_sbp"} and 30 <= v <= 300:
                field = "latest_sbp"
            elif item == "ebl" and 0 <= v <= 50000:
                x["max_ebl"] = max(x.get("max_ebl", v), v)
                continue
            else:
                continue
            if end - 15 <= t <= end and t >= x.get(field + "_time", -np.inf):
                x[field], x[field + "_time"] = v, t
    rows = []
    for opid in ops.op_id.astype("int64"):
        x = acc.get(int(opid), {})
        row = {"op_id": int(opid), **x}
        n = x.get("map_n", 0)
        row["map_below65_pct"] = x.get("map_below65_n", 0) / n if n else np.nan
        row["map_mean_deficit"] = x.get("map_deficit_sum", 0) / n if n else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def collect_subject_rows(path: Path, usecols: list[str], subjects: set[int],
                         predicate, time_col: str = "chart_time") -> dict[int, list[tuple]]:
    out: dict[int, list[tuple]] = defaultdict(list)
    for chunk in read_gzip_chunks(path, usecols):
        chunk["subject_id"] = pd.to_numeric(chunk.subject_id, errors="coerce")
        chunk = chunk[chunk.subject_id.isin(subjects)]
        if chunk.empty:
            continue
        chunk = chunk[predicate(chunk)]
        for r in chunk.itertuples(index=False, name=None):
            d = dict(zip(usecols, r))
            sid, t = _num(d.get("subject_id")), _num(d.get(time_col))
            if np.isfinite(sid) and np.isfinite(t):
                out[int(sid)].append((t, d))
    for sid in out:
        out[sid].sort(key=lambda z: z[0])
    return out


def collect_preop_vitals(ops: pd.DataFrame, path: Path) -> pd.DataFrame:
    subjects = set(pd.to_numeric(ops.subject_id, errors="coerce").dropna().astype("int64"))
    wanted = {"hr", "art_sbp", "nibp_sbp", "art_mbp", "nibp_mbp", "rr", "spo2",
              "bt", "fio2", "o2", "gcs_e", "gcs_m", "gcs_v"}
    rows_by_subject = collect_subject_rows(
        path, ["subject_id", "chart_time", "item_name", "value"], subjects,
        lambda c: c.item_name.isin(wanted))
    out = []
    for r in ops.itertuples(index=False):
        anchor = _num(getattr(r, "orin_time", np.nan))
        if not np.isfinite(anchor):
            anchor = _num(getattr(r, "opstart_time", np.nan))
        vals = {}
        sid = int(r.subject_id)
        for t, d in rows_by_subject.get(sid, []):
            if t > anchor:
                break
            if anchor - t > 24 * 60:
                continue
            item, v = d["item_name"], _num(d["value"])
            if not np.isfinite(v):
                continue
            key = {"art_sbp": "sbp", "nibp_sbp": "sbp",
                   "art_mbp": "map", "nibp_mbp": "map"}.get(item, item)
            if key not in vals or t >= vals[key + "_time"]:
                vals[key], vals[key + "_time"] = v, t
        out.append({"op_id": int(r.op_id), **{"preop_" + k: v for k, v in vals.items()}})
    return pd.DataFrame(out)


def collect_preop_labs(ops: pd.DataFrame, path: Path) -> pd.DataFrame:
    subjects = set(pd.to_numeric(ops.subject_id, errors="coerce").dropna().astype("int64"))
    wanted = {"creatinine", "hb", "troponin_i", "troponin_t"}
    rows_by_subject = collect_subject_rows(
        path, ["subject_id", "chart_time", "item_name", "value"], subjects,
        lambda c: c.item_name.isin(wanted))
    out = []
    for r in ops.itertuples(index=False):
        anchor = _num(getattr(r, "orin_time", np.nan))
        if not np.isfinite(anchor):
            anchor = _num(getattr(r, "opstart_time", np.nan))
        op_start = _num(getattr(r, "opstart_time", np.nan))
        if not np.isfinite(op_start):
            op_start = anchor
        op_end = _num(getattr(r, "opend_time", np.nan))
        if not np.isfinite(op_end):
            op_end = _num(getattr(r, "orout_time", np.nan))
        sid, values = int(r.subject_id), {}
        intraop_hb = []
        for t, d in rows_by_subject.get(sid, []):
            item, v = d["item_name"], _num(d["value"])
            if t <= anchor and anchor - t <= 30 * 24 * 60 and np.isfinite(v):
                key = "preop_" + item
                if key not in values or t >= values[key + "_time"]:
                    values[key], values[key + "_time"] = v, t
            if item == "hb" and np.isfinite(v) and op_start <= t <= op_end:
                intraop_hb.append(v)
        if intraop_hb:
            values["intraop_hb_min"] = float(min(intraop_hb))
        out.append({"op_id": int(r.op_id), **{k: v for k, v in values.items() if not k.endswith("_time")}})
    return pd.DataFrame(out)


def collect_sequence_baselines(dataset) -> pd.DataFrame:
    """Causal pre-OR baseline observations already present in the sequence."""
    vocab = dataset.meta["token_vocabulary"]
    wanted = {"baseline_spo2": vocab.get("context:baseline:spo2"),
              "baseline_hb_context": vocab.get("context:baseline:hemoglobin")}
    rows = []
    for ep in range(len(dataset.ptr) - 1):
        lo, hi = int(dataset.ptr[ep]), int(dataset.ptr[ep + 1])
        token = np.asarray(dataset.token_id[lo:hi])
        value = np.asarray(dataset.value[lo:hi])
        has = np.asarray(dataset.has_value[lo:hi])
        row = {"sequence_id": ep}
        for name, tid in wanted.items():
            if tid is None:
                continue
            ix = np.flatnonzero((token == tid) & (has > 0))
            if len(ix):
                row[name] = float(value[ix[-1]])
        rows.append(row)
    return pd.DataFrame(rows)


def collect_history(ops: pd.DataFrame, path: Path, medications_path: Path):
    subjects = set(pd.to_numeric(ops.subject_id, errors="coerce").dropna().astype("int64"))
    dx_rows = collect_subject_rows(
        path, ["subject_id", "chart_time", "icd10_cm"], subjects,
        lambda c: c.icd10_cm.notna())
    med_rows = collect_subject_rows(
        medications_path, ["subject_id", "chart_time", "drug_name", "atc_code"], subjects,
        lambda c: c.drug_name.astype(str).str.contains("insulin", case=False, na=False) |
                  c.atc_code.astype(str).str.upper().str.startswith("A10A"))
    history = []
    for r in ops.itertuples(index=False):
        anchor = _num(getattr(r, "opstart_time", np.nan))
        if not np.isfinite(anchor):
            anchor = _num(getattr(r, "orin_time", np.nan))
        sid = int(r.subject_id)
        codes = []
        recent_infection = False
        flags = {"diabetes": False, "chf": False, "ascites": False,
                 "hypertension": False, "ihd": False, "cerebrovascular": False}
        for t, d in dx_rows.get(sid, []):
            if t > anchor:
                break
            code = str(d["icd10_cm"]).upper().replace(".", "")
            if code:
                codes.append(code)
                recent_infection |= (anchor - t <= 30 * 24 * 60 and
                                     code[:3] in {f"J{i:02d}" for i in range(23)})
                flags["diabetes"] |= code.startswith(("E08", "E09", "E10", "E11", "E13"))
                flags["chf"] |= code.startswith("I50")
                flags["ascites"] |= code.startswith("R18")
                flags["hypertension"] |= code.startswith(("I10", "I11", "I12", "I13", "I15"))
                flags["ihd"] |= code.startswith(("I20", "I21", "I22", "I23", "I24", "I25"))
                flags["cerebrovascular"] |= code.startswith(("I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69")) or code.startswith("G45")
        insulin = any(anchor - t <= 30 * 24 * 60 for t, _ in med_rows.get(sid, []) if t <= anchor)
        history.append({"op_id": int(r.op_id), **flags,
                        "recent_respiratory_infection": recent_infection,
                        "insulin_recorded_30d": insulin})
    return pd.DataFrame(history)


def news2_partial(row):
    score = 0
    n = 0
    rr, spo2, sbp, hr, temp = (row.get("preop_rr"), row.get("preop_spo2"),
                               row.get("preop_sbp"), row.get("preop_hr"), row.get("preop_bt"))
    if np.isfinite(rr):
        n += 1
        score += 3 if rr <= 8 else 1 if rr <= 11 else 0 if rr <= 20 else 2 if rr <= 24 else 3
    if np.isfinite(spo2):
        n += 1
        score += 3 if spo2 <= 91 else 2 if spo2 <= 93 else 1 if spo2 <= 95 else 0
    if np.isfinite(sbp):
        n += 1
        score += 3 if sbp <= 90 else 2 if sbp <= 100 else 1 if sbp <= 110 else 3 if sbp >= 220 else 0
    if np.isfinite(hr):
        n += 1
        score += 3 if hr <= 40 else 1 if hr <= 50 else 0 if hr <= 90 else 1 if hr <= 110 else 2 if hr <= 130 else 3
    if np.isfinite(temp):
        n += 1
        score += 3 if temp <= 35 else 1 if temp <= 36 else 0 if temp <= 38 else 1 if temp <= 39 else 2
    return (float(score) if n >= 3 else np.nan, n)


def mews_partial(row):
    score, n = 0, 0
    rr, sbp, hr, temp = (row.get("preop_rr"), row.get("preop_sbp"),
                         row.get("preop_hr"), row.get("preop_bt"))
    if np.isfinite(rr):
        n += 1; score += 2 if rr <= 8 or rr >= 25 else 1 if rr <= 11 or rr >= 21 else 0
    if np.isfinite(sbp):
        n += 1; score += 3 if sbp <= 70 else 2 if sbp <= 80 else 1 if sbp <= 100 else 0 if sbp <= 199 else 2
    if np.isfinite(hr):
        n += 1; score += 2 if hr <= 40 or hr >= 130 else 1 if hr <= 50 or hr >= 110 else 0
    if np.isfinite(temp):
        n += 1; score += 2 if temp < 35 or temp >= 38.5 else 1 if temp < 36 else 0
    return (float(score) if n >= 3 else np.nan, n)


def derive_scores(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["preop_creatinine", "preop_hb"]:
        if col not in df:
            df[col] = np.nan
    df["shock_index"] = df.latest_hr / df.latest_sbp
    df["modified_shock_index"] = df.latest_hr / df.latest_map
    df["sas"] = [
        (3 if r.min_map >= 70 else 2 if r.min_map >= 55 else 1 if r.min_map >= 40 else 0) +
        (4 if r.min_hr <= 55 else 3 if r.min_hr <= 65 else 2 if r.min_hr <= 75 else 1 if r.min_hr <= 85 else 0) +
        (3 if r.max_ebl <= 100 else 2 if r.max_ebl <= 600 else 1 if r.max_ebl <= 1000 else 0)
        if all(np.isfinite(x) for x in [r.min_map, r.min_hr, r.max_ebl]) else np.nan
        for r in df.itertuples(index=False)
    ]
    # Adapted GS-AKI: preserve the published nine binary risk factors; recorded
    # diagnoses are treated as absent when no preoperative code was recorded.
    department = df.department.fillna("").astype(str).str.upper()
    pcs = df.icd10_pcs.fillna("").astype(str).str.upper()
    intraperitoneal_proxy = pcs.str.startswith(("0D", "0W")) | department.isin(["GS"])
    df["gs_aki_adapted"] = (
        (pd.to_numeric(df.age, errors="coerce") >= 56).astype(int) +
        (df.sex.astype(str).str.upper() == "M").astype(int) +
        (pd.to_numeric(df.emop, errors="coerce").fillna(0) > 0).astype(int) +
        intraperitoneal_proxy.astype(int) + df.diabetes.astype(int) + df.chf.astype(int) +
        df.ascites.astype(int) + df.hypertension.astype(int) +
        (pd.to_numeric(df.preop_creatinine, errors="coerce") >= 1.2).fillna(False).astype(int)
    )
    df["asa_plus_gs_aki"] = (pd.to_numeric(df.asa, errors="coerce") - 1) + df.gs_aki_adapted
    high_risk_rcri = (pcs.str.startswith(("0B", "0D")) |
                      department.isin(["CTS", "GS"]))
    df["rcri_proxy"] = (
        high_risk_rcri.astype(int) + df.ihd.astype(int) + df.chf.astype(int) +
        df.cerebrovascular.astype(int) + df.insulin_recorded_30d.astype(int) +
        (pd.to_numeric(df.preop_creatinine, errors="coerce") > 2.0).fillna(False).astype(int)
    )
    age = pd.to_numeric(df.age, errors="coerce")
    # ARISCAT point sum. Incision class and recent infection are necessarily
    # proxies in this source; unknown/missing inputs stay missing.
    spo2 = pd.to_numeric(df.preop_spo2, errors="coerce")
    hb = pd.to_numeric(df.preop_hb, errors="coerce")
    ar = pd.Series(0.0, index=df.index)
    ar += np.select([age > 80, age >= 51], [16, 3], default=0)
    ar += np.select([spo2 <= 90, spo2 <= 95], [24, 8], default=0)
    ar += df.recent_respiratory_infection.astype(int) * 17
    ar += (hb <= 10).fillna(False).astype(int) * 11
    ar += np.where(department.eq("CTS"), 24, np.where(department.eq("GS"), 15, 0))
    dur = pd.to_numeric(df.duration_min, errors="coerce")
    ar += np.select([dur >= 180, dur >= 120], [23, 16], default=0)
    ar += (pd.to_numeric(df.emop, errors="coerce").fillna(0) > 0).astype(int) * 8
    ar.loc[spo2.isna() | hb.isna() | dur.isna() | age.isna()] = np.nan
    df["ariscat_adapted"] = ar
    news = [news2_partial(r._asdict()) for r in df.itertuples(index=False)]
    mews = [mews_partial(r._asdict()) for r in df.itertuples(index=False)]
    df["news2_partial"] = [x[0] for x in news]
    df["news2_components"] = [x[1] for x in news]
    df["mews_partial"] = [x[0] for x in mews]
    df["mews_components"] = [x[1] for x in mews]
    rr = pd.to_numeric(df.preop_rr, errors="coerce")
    sbp = pd.to_numeric(df.preop_sbp, errors="coerce")
    df["qsofa_2component"] = ((rr >= 22).astype(float) + (sbp <= 100).astype(float)).where(rr.notna() & sbp.notna())
    df["map_current_risk"] = -pd.to_numeric(df.latest_map, errors="coerce")
    df["map_burden_risk"] = pd.to_numeric(df.map_below65_pct, errors="coerce")
    df["ebl_risk"] = pd.to_numeric(df.max_ebl, errors="coerce")
    df["hb_drop"] = (pd.to_numeric(df.preop_hb, errors="coerce") -
                     pd.to_numeric(df.intraop_hb_min, errors="coerce"))
    return df


def extract_landmark_predictions(dataset, checkpoint, device, batch_size: int):
    admissions = pd.read_csv(dataset.path / "admissions.csv")
    vocab = dataset.meta["token_vocabulary"]
    surgery_token = int(vocab["event:surgery_end"])
    window_by_ep: dict[int, list[int]] = defaultdict(list)
    for wi, ep in enumerate(dataset.window_admission):
        window_by_ep[int(ep)].append(int(wi))
    positions: dict[int, tuple[int, int, float]] = {}
    for ep in range(len(admissions)):
        lo, hi = int(dataset.ptr[ep]), int(dataset.ptr[ep + 1])
        tids = np.asarray(dataset.token_id[lo:hi])
        marks = np.flatnonzero(tids == surgery_token)
        if not len(marks):
            continue
        t = float(dataset.time_min[lo + int(marks[-1])])
        local_pos = int(np.searchsorted(dataset.time_min[lo:hi], t, side="right") - 1)
        covering = [wi for wi in window_by_ep.get(ep, [])
                    if dataset.window_start[wi] <= local_pos <
                    dataset.window_start[wi] + dataset.window_length[wi]]
        if covering:
            wi = min(covering, key=lambda x: dataset.window_start[x])
            positions[ep] = (wi, local_pos - int(dataset.window_start[wi]), t)
    selected = sorted({x[0] for x in positions.values()})
    by_window: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for ep, (wi, pos, t) in positions.items():
        by_window[wi].append((ep, pos, t))
    model = build_model(dataset, checkpoint, device)
    model.eval()
    pred = {}
    loader = DataLoader(Subset(dataset, selected), batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate_event_sequences,
                        pin_memory=device.type == "cuda")
    with torch.inference_mode():
        for bno, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
            out = model(batch)
            ws = selected[bno * batch_size:bno * batch_size + len(batch["token_id"])]
            for bi, wi in enumerate(ws):
                for ep, pos, t in by_window[wi]:
                    rec = {"landmark_min": t}
                    for hi, hours in enumerate(dataset.trajectory_horizons_hours):
                        rec[f"trajectory_{hours:g}h"] = torch.sigmoid(out.trajectory_logits[bi, pos, hi]).float().cpu().numpy()
                    if out.fine_hazard_logits is not None:
                        rec["fine"] = out.fine_hazard_logits[bi, pos].float().sigmoid().cpu().numpy()
                        rec["long"] = out.long_hazard_logits[bi, pos].float().sigmoid().cpu().numpy()
                        rec["tail_mu"] = out.tail_mu[bi, pos].float().cpu().numpy()
                        rec["tail_log_sigma"] = out.tail_log_sigma[bi, pos].float().cpu().numpy()
                    pred[ep] = rec
    return admissions, pred


def risk_at_horizon(rec: dict, outcome_idx: int, hours: float, trajectory_idx: dict[float, int]) -> float:
    if hours in trajectory_idx:
        return float(rec[f"trajectory_{hours:g}h"][outcome_idx])
    fine = rec.get("fine")
    long = rec.get("long")
    if fine is None or long is None:
        return np.nan
    survival = 1.0
    if hours <= 2:
        n = int(round(hours * 12))
        if n:
            survival *= float(np.prod(1.0 - fine[outcome_idx, :n]))
        return 1.0 - survival
    survival *= float(np.prod(1.0 - fine[outcome_idx]))
    if hours <= 24:
        n = int(round((hours - 2.0) * 2))
        survival *= float(np.prod(1.0 - long[outcome_idx, :n]))
        return 1.0 - survival
    survival *= float(np.prod(1.0 - long[outcome_idx]))
    mu = float(rec["tail_mu"][outcome_idx])
    sigma = max(float(np.exp(rec["tail_log_sigma"][outcome_idx])), 1e-6)
    dt = hours - 24.0
    cdf = float(ndtr((math.log(max(dt, 1e-6)) - mu) / sigma))
    return 1.0 - survival * (1.0 - cdf)


def collect_labels(dataset, admissions, pred, endpoint_specs):
    outv = dataset.meta["outcome_vocabulary"]
    stored = dataset.meta.get("stored_outcome_vocabulary", outv)
    remap = dataset.outcome_remap
    rows = []
    event_cache = {}
    for ep, p in pred.items():
        lo, hi = int(dataset.ptr[ep]), int(dataset.ptr[ep + 1])
        times = np.asarray(dataset.time_min[lo:hi])
        raw = np.asarray(dataset.outcome[lo:hi])
        mapped = np.full(raw.shape, -1, dtype=np.int16)
        valid = (raw >= 0) & (raw < len(remap))
        mapped[valid] = remap[raw[valid]]
        event_cache[ep] = (times, mapped)
    trajectory_index = {float(h): i for i, h in enumerate(dataset.trajectory_horizons_hours)}
    for ep, p in pred.items():
        a = admissions.iloc[ep]
        times, mapped = event_cache[ep]
        landmark = p["landmark_min"]
        follow = float(times[-1] - landmark)
        row = {"sequence_id": int(ep), "subject_id": str(a.subject_id),
               "landmark_min": landmark, "followup_min": follow}
        for name, hours, outcomes in endpoint_specs:
            target_idxs = [outv.index(x) for x in outcomes if x in outv]
            if not target_idxs:
                row[f"label_{name}"] = np.nan
                row[f"maomao_{name}"] = np.nan
                continue
            mask = (times > landmark) & np.isin(mapped, target_idxs)
            if hours is None:
                y = int(mask.any())
                risk_parts = [risk_at_horizon(p, idx, 720.0, trajectory_index) for idx in target_idxs]
                row[f"label_{name}"] = y
                row[f"maomao_{name}"] = float(np.nanmax(risk_parts))
                continue
            horizon_min = hours * 60.0
            positive = bool(np.any(mask & (times <= landmark + horizon_min)))
            complete = follow >= horizon_min
            if name.startswith("inhospital_death"):
                # A live discharge before the fixed horizon is a known negative
                # for in-hospital death, even without postdischarge vital-status follow-up.
                discharge = _num(getattr(a, "discharge_time", np.nan))
                surgery_end = _num(getattr(a, "opend_time", np.nan))
                if np.isfinite(discharge) and np.isfinite(surgery_end) and discharge >= surgery_end:
                    complete = True
            row[f"label_{name}"] = 1 if positive else (0 if complete else np.nan)
            # Max constituent event risk is a rank-only composite, avoiding an
            # unsupported conditional-independence assumption for stage 2/3.
            risk_parts = [risk_at_horizon(p, idx, hours, trajectory_index) for idx in target_idxs]
            row[f"maomao_{name}"] = float(np.nanmax(risk_parts))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_metrics(y, preds, patient, repeats=300, seed=4251):
    unique = np.unique(patient)
    groups = {s: np.flatnonzero(patient == s) for s in unique}
    rng = np.random.default_rng(seed)
    names = list(preds)
    point = {n: (float(roc_auc_score(y, preds[n])), float(average_precision_score(y, preds[n]))) for n in names}
    vals = {n: [[], []] for n in names}
    diffs = {n: [[], []] for n in names if n != "MAOMAO"}
    for _ in range(repeats):
        draw = rng.choice(unique, size=len(unique), replace=True)
        ix = np.concatenate([groups[s] for s in draw])
        yy = y[ix]
        if yy.min() == yy.max():
            continue
        for name in names:
            auc = roc_auc_score(yy, preds[name][ix])
            ap = average_precision_score(yy, preds[name][ix])
            vals[name][0].append(auc); vals[name][1].append(ap)
        for name in diffs:
            for j in range(2):
                diffs[name][j].append(vals[name][j][-1] - vals["MAOMAO"][j][-1])
    result = {}
    for name in names:
        result[name] = {
            "auroc": {"estimate": point[name][0], "ci95": np.quantile(vals[name][0], [0.025, 0.975]).tolist()},
            "auprc": {"estimate": point[name][1], "ci95": np.quantile(vals[name][1], [0.025, 0.975]).tolist()},
        }
    result["score_minus_maomao"] = {
        name: {
            "auroc": {"estimate": point[name][0] - point["MAOMAO"][0], "ci95": np.quantile(diffs[name][0], [0.025, 0.975]).tolist()},
            "auprc": {"estimate": point[name][1] - point["MAOMAO"][1], "ci95": np.quantile(diffs[name][1], [0.025, 0.975]).tolist()},
        } for name in diffs
    }
    return result


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = checkpoint["args"]
    dataset = EventSequenceDataset(args.data_dir, cfg.get("block_size", 256),
                                   cfg.get("window_stride", 128), dynamic_windows=False)
    if dataset.split != "test":
        raise ValueError("Expanded comparison requires the independent sealed test directory")
    admissions, predictions = extract_landmark_predictions(dataset, checkpoint, device, args.batch_size)
    base = admissions.copy()
    base["sequence_id"] = np.arange(len(base))
    base = base.merge(collect_sequence_baselines(dataset), on="sequence_id", how="left", validate="one_to_one")
    base = base.merge(collect_or_inputs(base, args.vitals), on="op_id", how="left", validate="one_to_one")
    base = base.merge(collect_preop_vitals(base, args.ward_vitals), on="op_id", how="left", validate="one_to_one")
    base = base.merge(collect_preop_labs(base, args.labs), on="op_id", how="left", validate="one_to_one")
    base = base.merge(collect_history(base, args.diagnosis, args.medications), on="op_id", how="left", validate="one_to_one")
    if "baseline_spo2" in base:
        base["preop_spo2"] = pd.to_numeric(base.get("preop_spo2"), errors="coerce").fillna(base.baseline_spo2)
    base["duration_min"] = pd.to_numeric(base.opend_time, errors="coerce") - pd.to_numeric(base.opstart_time, errors="coerce")
    base = derive_scores(base)

    endpoint_specs = [
        ("hypotension_60m", 1.0, ["severe_map_hypotension"]),
        ("hypotension_6h", 6.0, ["severe_map_hypotension"]),
        ("pressor_1h", 1.0, ["vasopressor_start"]),
        ("pressor_6h", 6.0, ["vasopressor_start"]),
        ("icu_24h", 24.0, ["icu_transfer"]),
        ("crrt_24h", 24.0, ["crrt_start"]),
        ("ventilation_24h", 24.0, ["ventilation_start"]),
        ("rbc_6h", 6.0, ["rbc_transfusion"]),
        ("rbc_24h", 24.0, ["rbc_transfusion"]),
        ("troponin_elevation_24h", 24.0, ["troponin_i_elevation"]),
    ]
    labels = collect_labels(dataset, admissions, predictions, endpoint_specs)
    merged = base.merge(labels.drop(columns=["subject_id", "landmark_min", "followup_min"]),
                        on="sequence_id", how="inner", validate="one_to_one")
    score_map = {
        "SAS": ("sas", -1), "ASA-PS": ("asa", 1),
        "adapted GS-AKI": ("gs_aki_adapted", 1), "ASA + adapted GS-AKI": ("asa_plus_gs_aki", 1),
        "RCRI proxy": ("rcri_proxy", 1),
        "ARISCAT adapted": ("ariscat_adapted", 1),
        "NEWS2 physiologic subtotal": ("news2_partial", 1),
        "MEWS physiologic subtotal": ("mews_partial", 1),
        "qSOFA 2-component": ("qsofa_2component", 1),
        "Shock Index": ("shock_index", 1), "Modified Shock Index": ("modified_shock_index", 1),
        "Current MAP": ("map_current_risk", 1), "MAP burden <65": ("map_burden_risk", 1),
        "EBL": ("ebl_risk", 1), "Hb drop": ("hb_drop", 1),
    }
    endpoint_scores = {
        "hypotension_60m": ["Shock Index", "Modified Shock Index", "Current MAP", "MAP burden <65"],
        "hypotension_6h": ["Shock Index", "Modified Shock Index", "Current MAP", "MAP burden <65"],
        "pressor_1h": ["Shock Index", "Modified Shock Index", "Current MAP", "MAP burden <65"],
        "pressor_6h": ["Shock Index", "Modified Shock Index", "Current MAP", "MAP burden <65"],
        "icu_24h": ["SAS", "ASA-PS", "NEWS2 physiologic subtotal", "MEWS physiologic subtotal", "qSOFA 2-component"],
        "crrt_24h": ["ASA-PS", "adapted GS-AKI", "ASA + adapted GS-AKI"],
        "ventilation_24h": ["ASA-PS", "ARISCAT adapted", "NEWS2 physiologic subtotal", "MEWS physiologic subtotal", "qSOFA 2-component"],
        "rbc_6h": ["SAS", "EBL", "Hb drop", "Shock Index", "Modified Shock Index"],
        "rbc_24h": ["SAS", "EBL", "Hb drop", "Shock Index", "Modified Shock Index"],
        "troponin_elevation_24h": ["RCRI proxy", "Current MAP", "MAP burden <65"],
    }
    summary = {"status": "sealed_independent_test", "checkpoint": str(args.checkpoint.resolve()),
               "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
               "patient_overlap_with_development": 0,
               "landmark": "last token at recorded surgery_end; all score inputs restricted to information available by surgery end, preoperative components restricted to before OR entry/start",
               "score_definitions": {
                   "SAS": "published 0-10 score: intraoperative minimum MAP (0-3), minimum HR (0-4), and EBL (0-3); higher is lower risk",
                   "GS-AKI": "nine-factor adaptation: age>=56, male, emergency, intraperitoneal procedure proxy, diabetes, CHF, ascites, hypertension, preoperative creatinine>=1.2; prior recorded ICD codes are treated as absent if not recorded",
                   "RCRI": "six binary Lee-style components, with high-risk surgery classified using PCS/departments; comorbidities/insulin from preoperative recorded codes/medications",
                   "ARISCAT": "point sum with age, preoperative SpO2/Hb, infection code in preceding 30d, duration, binary emergency and department-based incision proxy; oxygen/respiratory infection/invasion definitions incomplete",
                   "NEWS2/MEWS/qSOFA": "partial preoperative physiologic subtotals from latest available ward observations in prior 24h; consciousness and oxygen treatment are not included in NEWS2/MEWS; qSOFA has only SBP and RR",
                   "MAP burden": "proportion of recorded intraoperative MAP readings <65 mmHg; this is reading-weighted rather than duration-weighted",
                   "AKI": "event-sequence AKI stage 2 or 3 signal; creatinine/urine-output KDIGO adjudication not independently reconstructed",
               },
               "endpoint_limits": {
                   "icu_24h": "ICU transfer may be planned",
                   "pressor": "medication/vasopressor documentation start, not adjudicated shock",
                   "hypotension": "threshold signal from monitored MAP; not independently adjudicated postoperative complication",
                   "ventilation": "recorded ventilation-start transition, not separately adjudicated reintubation",
                   "troponin_elevation_24h": "any troponin-I elevation signal, not adjudicated MINS or RCRI original MACE",
                   "rbc": "recorded RBC transfusion start signal; amount and indication are not adjudicated",
               }, "comparisons_by_endpoint": {}}
    for endpoint, scores in endpoint_scores.items():
        ycol, hcol = f"label_{endpoint}", f"maomao_{endpoint}"
        out_rows = []
        for score_name in scores:
            score_col, direction = score_map[score_name]
            part = merged.loc[merged[ycol].notna() & merged[hcol].notna() & merged[score_col].notna()].copy()
            if part[ycol].nunique() < 2 or len(part) < 30:
                continue
            y = part[ycol].to_numpy(dtype=np.int8)
            preds = {score_name: direction * pd.to_numeric(part[score_col]).to_numpy(dtype=float),
                     "MAOMAO": pd.to_numeric(part[hcol]).to_numpy(dtype=float)}
            metrics = paired_metrics(y, preds, part.subject_id.astype(str).to_numpy(),
                                     args.bootstrap_repeats,
                                     args.bootstrap_seed + len(summary["comparisons_by_endpoint"]) * 17 + len(out_rows))
            out_rows.append({"comparator": score_name, "n_episodes": len(part),
                             "n_patients": int(part.subject_id.nunique()), "events": int(y.sum()),
                             "event_rate": float(y.mean()), "score_column": score_col,
                             "metrics_patient_cluster_bootstrap": metrics})
        summary["comparisons_by_endpoint"][endpoint] = out_rows
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / "expanded_test_episode_predictions.csv", index=False)
    (args.output_dir / "expanded_comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "comparisons_by_endpoint"}, ensure_ascii=False, indent=2))
    for endpoint, groups in summary["comparisons_by_endpoint"].items():
        print(endpoint, json.dumps(groups, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=ROOT / "outputs/classic_score_comparison/independent_15pct_test/test")
    ap.add_argument("--checkpoint", type=Path, default=ROOT / "outputs/classic_score_comparison/independent_15pct_test/maomao_fresh/best_model.pt")
    ap.add_argument("--vitals", type=Path, default=ROOT / "data/train_data/vitals.csv.gz")
    ap.add_argument("--ward-vitals", type=Path, default=ROOT / "data/train_data/ward_vitals.csv.gz")
    ap.add_argument("--labs", type=Path, default=ROOT / "data/train_data/labs.csv.gz")
    ap.add_argument("--diagnosis", type=Path, default=ROOT / "data/train_data/diagnosis.csv.gz")
    ap.add_argument("--medications", type=Path, default=ROOT / "data/train_data/medications.csv.gz")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "outputs/classic_score_comparison/independent_15pct_test/expanded_results")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--bootstrap-repeats", type=int, default=300)
    ap.add_argument("--bootstrap-seed", type=int, default=73041)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
