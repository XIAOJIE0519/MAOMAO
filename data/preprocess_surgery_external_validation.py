#!/usr/bin/env python3
"""Screen surgical anesthesia records, clean supported physiology, and freeze MAOMAO inputs.

The adapter selects records from documented surgical anesthesia cohorts,
confirmed ASAC operations and the already
quality-controlled NTUH surgery ECG cohort. It never promotes an ICU/ED record
to a surgery record. ECG is reduced to minute-level heart rate estimates; raw
500-Hz points are never tokenized.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sqlite3
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import h5py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.external_validation_common import ValidationStore, write_validation_dataset


ROOT = Path(__file__).resolve().parents[1]
PART1 = ROOT / "data/surgery_part1/surgery"
PART2 = ROOT / "data/surgery_part2/processed_surgical_ecg"
TRAIN = ROOT / "data/perioperative_event_sequences_v5_richctx_static7"
OUT = ROOT / "outputs/surgery_part1_part2_maomao_validation"
UQ_ZIP = PART1 / "UQ_Vital_Signs_32/uqvitalsignsdata_case01to32.zip"
ASAC_ZIP = PART1 / "Auckland_Anesthetic_Records_34/ASAC_dataset.zip"
EDS_ZIP = PART1 / "Auckland_Anesthetic_Records_34/EDS_MH_data.zip"
ECG_DIR = PART2 / "ecg_float32_500hz_bandpass_0p5_40hz"
HR_FEATURES = {
    "HR": ("hr", 25, 240), "Pulse": ("hr", 25, 240), "PULSE": ("hr", 25, 240),
    "SpO2": ("spo2", 50, 100), "ETCO2": ("etco2", 10, 100),
    "etCO2": ("etco2", 10, 100), "AWRR": ("rr", 1, 80),
    "awRR": ("rr", 1, 80), "RR": ("rr", 1, 80), "FV_RR": ("rr", 1, 80),
    "T1": ("bt", 25, 45), "T2": ("bt", 25, 45), "Temp": ("bt", 25, 45),
    "ART min": ("art_dbp", 20, 200), "ART max": ("art_sbp", 30, 300),
    "ART mean": ("art_mbp", 20, 250), "NBP min": ("nibp_dbp", 20, 200),
    "NBP max": ("nibp_sbp", 30, 300), "NBP mean": ("nibp_mbp", 20, 250),
    "NBP (Sys)": ("nibp_sbp", 30, 300), "NBP (Dia)": ("nibp_dbp", 20, 200),
    "NBP (Mean)": ("nibp_mbp", 20, 250), "ART (Sys)": ("art_sbp", 30, 300),
    "ART (Dia)": ("art_dbp", 20, 200), "ART (Mean)": ("art_mbp", 20, 250),
}


def add_clean_rows(store: ValidationStore, ep: int, frame: pd.DataFrame) -> None:
    rows = []
    for row in frame.itertuples(index=False):
        feature = row.feature
        t = float(row.time_min)
        value = float(row.value)
        if math.isfinite(t) and math.isfinite(value) and t >= 0:
            rows.append((ep, int(t // 5.0), f"vitals:{feature}", value, 1))
    store.add_observations(rows)


def map_measurement(name: str, values: np.ndarray) -> tuple[str | None, np.ndarray]:
    spec = HR_FEATURES.get(name)
    if spec is None:
        return None, np.empty(0, dtype=np.float32)
    feature, low, high = spec
    values = np.asarray(values, dtype=np.float64)
    values[(values < low) | (values > high) | ~np.isfinite(values)] = np.nan
    return feature, values


def bin_measurements(ep: int, times_min: np.ndarray, signals: dict[str, np.ndarray]) -> pd.DataFrame:
    pieces = []
    for feature, values in signals.items():
        values = np.asarray(values, dtype=float)
        valid = np.isfinite(times_min) & np.isfinite(values) & (times_min >= 0)
        if not valid.any():
            continue
        bins = (times_min[valid] // 5).astype(int)
        temp = pd.DataFrame({"bin": bins, "value": values[valid]}).groupby("bin").value.mean()
        pieces.extend((ep, int(b), (int(b) + 0.5) * 5.0, feature, float(v))
                      for b, v in temp.items())
    return pd.DataFrame(pieces, columns=["ep_idx", "bin", "time_min", "feature", "value"])


def parse_uq(store: ValidationStore, cleaned_path: Path) -> tuple[list[dict], list[dict], dict]:
    episodes, manifest, clean = [], [], []
    source_audit = {"source": "UQ Vital Signs 32", "selected_cases": 0,
                    "excluded_non_surgical": 0, "outside_plausible_range": Counter(),
                    "invalid_time_rows": 0, "five_minute_measurements": 0,
                    "phase_summaries_available": False}
    with ZipFile(UQ_ZIP) as archive:
        members = sorted(n for n in archive.namelist()
                         if n.endswith("_trenddata.csv") and "/case" in n)
        for ep, member in enumerate(members):
            case = Path(member).parent.name
            # These 32 numbered records are the documented UQ surgical cohort;
            # select this cohort before reading/cleaning its physiology.
            if not case.startswith("case") or not case[4:].isdigit():
                source_audit["excluded_non_surgical"] += 1
                continue
            # Some monitor alarm strings contain unquoted commas after the
            # numeric columns. Parse rows first and keep the fixed header width.
            with io.TextIOWrapper(archive.open(member), encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream)
                header = next(reader)
                raw_rows = [row[:len(header)] + [""] * max(0, len(header) - len(row))
                            for row in reader]
            frame = pd.DataFrame(raw_rows, columns=header)
            rel = pd.to_numeric(frame.get("RelativeTimeMilliseconds"), errors="coerce")
            time_min = rel.to_numpy(float) / 60000.0
            valid_time = np.isfinite(time_min) & (time_min >= 0)
            source_audit["invalid_time_rows"] += int((~valid_time).sum())
            signals = {}
            rejected = Counter()
            for column, (feature, low, high) in HR_FEATURES.items():
                if column not in frame:
                    continue
                values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
                original_finite = np.isfinite(values)
                plausible = original_finite & (values >= low) & (values <= high)
                rejected[column] += int((original_finite & ~plausible).sum())
                if plausible.any():
                    signals.setdefault(feature, []).append((time_min[plausible], values[plausible]))
            source_audit["outside_plausible_range"].update(rejected)
            merged = {}
            for feature, parts in signals.items():
                merged[feature] = np.concatenate([x[1] for x in parts])
                times = np.concatenate([x[0] for x in parts])
                temp = bin_measurements(ep, times, {feature: merged[feature]})
                clean.append(temp)
            per_case = pd.concat([x for x in clean if len(x) and x.ep_idx.iloc[0] == ep],
                                 ignore_index=True) if any(len(x) and x.ep_idx.iloc[0] == ep for x in clean) else pd.DataFrame()
            if per_case.empty:
                manifest.append({"source_id": case, "status": "excluded_qc", "reason": "no plausible supported vitals"})
                continue
            duration = float(np.nanmax(time_min[valid_time]))
            episodes.append({"ep_idx": ep, "admission_id": f"uq:{case}", "subject_id": f"uq:{case}",
                             "age": np.nan, "male": np.nan, "weight_kg": np.nan, "height_cm": np.nan,
                             "asa": np.nan, "emop": np.nan, "antype": None, "department": "UQ surgery",
                             "duration_min": max(5.0, duration), "or_in_min": 0.0, "an_start_min": 0.0,
                             "surgery_start_min": np.nan, "surgery_end_min": np.nan,
                             "an_end_min": np.nan, "or_out_min": max(5.0, duration),
                             "icu_in_min": np.nan, "icu_out_min": np.nan, "death_min": np.nan,
                             "phase_events_available": False})
            add_clean_rows(store, ep, per_case)
            source_audit["selected_cases"] += 1
            source_audit["five_minute_measurements"] += len(per_case)
            manifest.append({"source_id": case, "source_entry": member, "status": "included",
                             "surgery_screen": "documented UQ surgical anesthesia cohort, case01-case32",
                             "raw_rows": len(frame), "clean_rows": len(per_case),
                             "duration_min": round(duration, 2),
                             "measurement_columns": sorted(signals),
                             "out_of_range_values": dict(rejected)})
    cleaned = pd.concat(clean, ignore_index=True) if clean else pd.DataFrame()
    cleaned.to_csv(cleaned_path, index=False)
    source_audit["outside_plausible_range"] = dict(source_audit["outside_plausible_range"])
    return episodes, manifest, source_audit


def parse_asac(store: ValidationStore, cleaned_path: Path) -> tuple[list[dict], list[dict], dict]:
    episodes, manifest, clean = [], [], []
    source_audit = {"source": "Auckland ASAC", "selected_cases": 0,
                    "excluded_non_surgical": 0, "five_minute_measurements": 0,
                    "invalid_or_out_of_range_values": 0,
                    "phase_summaries_available": True}
    with ZipFile(ASAC_ZIP) as archive:
        members = sorted(n for n in archive.namelist() if n.endswith(".xml"))
        for ep, member in enumerate(members):
            root = ET.fromstring(archive.read(member))
            op = (root.findtext("./operation/opdescription") or "").strip()
            if not op or op.lower() in {"unknown", "none", "n/a"}:
                source_audit["excluded_non_surgical"] += 1
                manifest.append({"source_id": Path(member).stem, "status": "excluded_screen",
                                 "reason": "operation description does not identify a surgery"})
                continue
            event_data = []
            or_arrivals = []
            for event in root.findall("./events/event"):
                label = (event.findtext("evdescription") or "").strip().lower()
                event_time = pd.to_numeric(event.findtext("evtime"), errors="coerce")
                if np.isfinite(event_time):
                    event_data.append((float(event_time), label))
                    if "arriv" in label and "or" in label:
                        or_arrivals.append(float(event_time))
            time_shift_s = -min(or_arrivals) if or_arrivals and min(or_arrivals) < 0 else 0.0
            vars_by_feature: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
            max_time_s = 0.0
            for var in root.findall("./data/var"):
                name = (var.findtext("vaname") or "").strip()
                if name not in HR_FEATURES:
                    continue
                feature, low, high = HR_FEATURES[name]
                ts = np.asarray([pd.to_numeric(x, errors="coerce") for x in
                                 (var.findtext("vatimes") or "").split(",")], dtype=float)
                vs = np.asarray([pd.to_numeric(x, errors="coerce") for x in
                                 (var.findtext("vavalues") or "").split(",")], dtype=float)
                n = min(len(ts), len(vs))
                ts, vs = ts[:n], vs[:n]
                good = np.isfinite(ts) & np.isfinite(vs) & (ts >= 0) & (vs >= low) & (vs <= high)
                source_audit["invalid_or_out_of_range_values"] += int(n - good.sum())
                if good.any():
                    max_time_s = max(max_time_s, float(ts[good].max()) + time_shift_s)
                    vars_by_feature[feature].append(((ts[good] + time_shift_s) / 60.0, vs[good]))
            signals, times_by_feature = {}, {}
            for feature, parts in vars_by_feature.items():
                times_by_feature[feature] = np.concatenate([p[0] for p in parts])
                signals[feature] = np.concatenate([p[1] for p in parts])
            cleaned = bin_measurements(ep, np.asarray([0.0]), {})
            rows = []
            for feature, values in signals.items():
                rows.append(bin_measurements(ep, times_by_feature[feature], {feature: values}))
            case_clean = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
            if case_clean.empty:
                manifest.append({"source_id": Path(member).stem, "status": "excluded_qc",
                                 "reason": "no plausible supported physiology"})
                continue
            patient = root.find("patient")
            get_number = lambda tag: pd.to_numeric(patient.findtext(tag) if patient is not None else None,
                                                    errors="coerce")
            age, weight, height = get_number("age"), get_number("weight"), get_number("height")
            sex = (patient.findtext("sex") if patient is not None else "") or ""
            surgery_start, surgery_end, or_out, an_end = np.nan, np.nan, np.nan, np.nan
            for event_time, label in event_data:
                t_min = (event_time + time_shift_s) / 60.0
                if any(x in label for x in ("knife to skin", "surgery start", "surgery begins")):
                    surgery_start = max(0.0, t_min)
                if any(x in label for x in ("surgery over", "surgery end", "surgery finished")):
                    surgery_end = max(0.0, t_min)
                if any(x in label for x in ("patient leaves or", "patient exits or", "leave or", "exits or")):
                    or_out = max(0.0, t_min)
                if any(x in label for x in ("anaesthesia ends", "anesthesia ends", "anaesthetic ends", "anesthetic ends")):
                    an_end = max(0.0, t_min)
            duration = max(max_time_s / 60.0, float(surgery_end) if np.isfinite(surgery_end) else 0.0, 5.0)
            asa = pd.to_numeric(patient.findtext("asa") if patient is not None else None, errors="coerce")
            ep_row = {"ep_idx": ep, "admission_id": f"asac:{Path(member).stem}",
                      "subject_id": f"asac:{Path(member).stem}", "age": age,
                      "male": 1.0 if sex.strip().lower() in {"m", "male"} else (0.0 if sex.strip().lower() in {"f", "female"} else np.nan),
                      "weight_kg": weight, "height_cm": height, "asa": asa, "emop": 0,
                      "antype": "General", "department": "Auckland surgery",
                      "duration_min": duration, "or_in_min": 0.0 if or_arrivals else np.nan,
                      "an_start_min": np.nan,
                      "surgery_start_min": surgery_start, "surgery_end_min": surgery_end,
                      "an_end_min": an_end, "or_out_min": or_out,
                      "icu_in_min": np.nan, "icu_out_min": np.nan, "death_min": np.nan}
            ep_row["phase_events_available"] = True
            episodes.append(ep_row)
            add_clean_rows(store, ep, case_clean)
            clean.append(case_clean)
            source_audit["selected_cases"] += 1
            source_audit["five_minute_measurements"] += len(case_clean)
            manifest.append({"source_id": Path(member).stem, "source_entry": member,
                             "status": "included", "operation_description": op,
                             "surgery_screen": "non-empty documented operation description",
                             "clean_rows": len(case_clean), "duration_min": round(duration, 2),
                             "measurement_features": sorted(signals)})
    # EDS_MH files often say "unknown" for the operation field. Include only
    # records whose time-stamped event explicitly documents that surgery began.
    eds_maps = {
        "ecg.hr": "hr", "nibp.hr": "hr", "p1.hr": "hr",
        "spo2.SpO2": "spo2", "co2.et": "etco2", "co2.rr": "rr",
        "nibp.sys": "nibp_sbp", "nibp.dia": "nibp_dbp", "nibp.mean": "nibp_mbp",
        "p1.sys": "art_sbp", "p1.dia": "art_dbp", "p1.mean": "art_mbp",
        "t1.temp": "bt",
    }
    with ZipFile(EDS_ZIP) as archive:
        for member in sorted(n for n in archive.namelist() if n.endswith(".xml")):
            root = ET.fromstring(archive.read(member))
            events = root.findall("./events/event")
            surgical_events = []
            for event in events:
                label = (event.findtext("evdescription") or "").strip().lower()
                if any(term in label for term in ("surgery started", "surgery start", "surgery restarts")):
                    day_fraction = pd.to_numeric(event.findtext("evtime"), errors="coerce")
                    if np.isfinite(day_fraction):
                        surgical_events.append((float(day_fraction) * 86400.0, label))
            if not surgical_events:
                source_audit["excluded_non_surgical"] += 1
                manifest.append({"source_id": Path(member).stem, "status": "excluded_screen",
                                 "reason": "operation unknown and no explicit surgery-start event"})
                continue
            ep = len(episodes)
            observations = defaultdict(list)
            all_times = []
            rejected = 0
            for var in root.findall("./data/var"):
                key = (var.findtext("vaname") or "").strip()
                feature = eds_maps.get(key)
                if feature is None:
                    continue
                raw_times = (var.findtext("vatimes") or "").split(",")
                raw_values = (var.findtext("vavalues") or "").split(",")
                times, vals = [], []
                for t_text, v_text in zip(raw_times, raw_values):
                    try:
                        h, m, s = (int(part) for part in t_text.strip().split(":"))
                        t_sec = h * 3600 + m * 60 + s
                        value = float(v_text)
                    except (ValueError, TypeError):
                        rejected += 1
                        continue
                    limits = {"hr": (25, 240), "spo2": (50, 100), "etco2": (10, 100),
                              "rr": (1, 80), "nibp_sbp": (30, 300), "nibp_dbp": (20, 200),
                              "nibp_mbp": (20, 250), "art_sbp": (30, 300), "art_dbp": (20, 200),
                              "art_mbp": (20, 250), "bt": (25, 45)}[feature]
                    if not np.isfinite(value) or not limits[0] <= value <= limits[1]:
                        rejected += 1
                        continue
                    times.append(t_sec); vals.append(value); all_times.append(t_sec)
                if times:
                    observations[feature].append((np.asarray(times, float), np.asarray(vals, float)))
            if not all_times:
                manifest.append({"source_id": Path(member).stem, "status": "excluded_qc",
                                 "reason": "no plausible supported physiology"})
                continue
            origin = min(all_times)
            start_sec = min(x[0] for x in surgical_events)
            duration = (max(all_times) - origin) / 60.0
            surgery_start = max(0.0, (start_sec - origin) / 60.0)
            rows = []
            for feature, parts in observations.items():
                times = np.concatenate([part[0] for part in parts])
                vals = np.concatenate([part[1] for part in parts])
                rows.append(bin_measurements(ep, (times - origin) / 60.0, {feature: vals}))
            case_clean = pd.concat(rows, ignore_index=True)
            patient = root.find("patient")
            def eds_number(tag):
                val = pd.to_numeric(patient.findtext(tag) if patient is not None else None, errors="coerce")
                return float(val) if np.isfinite(val) and val > 0 else np.nan
            sex = ((patient.findtext("sex") if patient is not None else "") or "").strip().lower()
            episode = {"ep_idx": ep, "admission_id": f"eds:{Path(member).stem}",
                       "subject_id": f"eds:{Path(member).stem}", "age": eds_number("age"),
                       "male": 1.0 if sex in {"m", "male"} else (0.0 if sex in {"f", "female"} else np.nan),
                       "weight_kg": eds_number("weight"), "height_cm": eds_number("height"),
                       "asa": eds_number("asa"), "emop": 0, "antype": "General",
                       "department": "Auckland surgery", "duration_min": max(5.0, duration),
                       "or_in_min": np.nan, "an_start_min": np.nan, "surgery_start_min": surgery_start,
                       "surgery_end_min": np.nan, "an_end_min": np.nan,
                       "or_out_min": np.nan, "icu_in_min": np.nan,
                       "icu_out_min": np.nan, "death_min": np.nan,
                       "phase_events_available": True}
            episodes.append(episode); add_clean_rows(store, ep, case_clean); clean.append(case_clean)
            source_audit["selected_cases"] += 1
            source_audit["five_minute_measurements"] += len(case_clean)
            source_audit["invalid_or_out_of_range_values"] += rejected
            manifest.append({"source_id": Path(member).stem, "status": "included",
                             "operation_description": "unknown; explicit surgery-start event",
                             "surgery_screen": "time-stamped surgery-start event",
                             "clean_rows": len(case_clean), "duration_min": round(duration, 2),
                             "surgery_start_min": round(surgery_start, 2),
                             "measurement_features": sorted(observations),
                             "invalid_or_out_of_range_values": rejected})
    pd.concat(clean, ignore_index=True).to_csv(cleaned_path, index=False) if clean else pd.DataFrame().to_csv(cleaned_path, index=False)
    return episodes, manifest, source_audit


def detect_hr_minute(signal: np.ndarray, fs: float = 500.0) -> tuple[np.ndarray, dict]:
    minute = int(fs * 60)
    n_minutes = len(signal) // minute
    values, qc = [], {"minute_windows": n_minutes, "valid_minute_windows": 0,
                      "invalid_minute_windows": 0, "polarity_disagreement_windows": 0}
    for index in range(n_minutes):
        x = np.asarray(signal[index * minute:(index + 1) * minute], dtype=np.float64)
        estimates = []
        for polarity in (1.0, -1.0):
            y = polarity * x
            center = np.median(y)
            mad = np.median(np.abs(y - center)) * 1.4826
            prominence = max(0.25, 2.5 * mad)
            peaks, _ = find_peaks(y, distance=int(fs * 0.30), prominence=prominence)
            rr = np.diff(peaks) / fs
            rr = rr[(rr >= 0.30) & (rr <= 2.40)]
            if len(rr) >= 8:
                hr = 60.0 / float(np.median(rr))
                if 25.0 <= hr <= 200.0:
                    estimates.append((float(np.median(np.abs(rr - np.median(rr)))), hr))
        if not estimates:
            values.append(np.nan); qc["invalid_minute_windows"] += 1
            continue
        estimates.sort(key=lambda x: x[0])
        if len(estimates) == 2 and abs(estimates[0][1] - estimates[1][1]) > 15:
            qc["polarity_disagreement_windows"] += 1
        values.append(estimates[0][1]); qc["valid_minute_windows"] += 1
    return np.asarray(values, dtype=np.float32), qc


def parse_ntuh(store: ValidationStore, cleaned_path: Path) -> tuple[list[dict], list[dict], dict]:
    episodes, manifest, clean = [], [], []
    source_audit = {"source": "NTUH Raw Data 110", "selected_surgical_ecg": 0,
                    "excluded_after_prior_qc": 0, "five_minute_measurements": 0,
                    "heart_rate_derivation": "500-Hz ECG; polarity-aware QRS peak intervals, 1-min HR then 5-min median",
                    "phase_summaries_available": False}
    case_manifest = pd.read_csv(PART2 / "case_manifest.csv")
    for ep, row in enumerate(case_manifest.itertuples(index=False)):
        if row.status != "included":
            source_audit["excluded_after_prior_qc"] += 1
            manifest.append({"source_id": row.record_id, "status": "excluded_prior_qc", "reason": row.qc_message})
            continue
        path = PART2 / row.cleaned_signal
        sig = np.load(path, mmap_mode="r")
        hr_min, hr_qc = detect_hr_minute(sig)
        if not len(hr_min) or hr_qc["valid_minute_windows"] / max(1, hr_qc["minute_windows"]) < 0.5:
            manifest.append({"source_id": row.record_id, "status": "excluded_hr_qc", **hr_qc})
            source_audit["excluded_after_prior_qc"] += 1
            continue
        duration = len(sig) / 500.0 / 60.0
        time_min = np.arange(len(hr_min), dtype=float)
        rows = bin_measurements(ep, time_min, {"hr": hr_min})
        rows = rows.dropna(subset=["value"])
        episodes.append({"ep_idx": ep, "admission_id": f"ntuh:{row.record_id}",
                         "subject_id": f"ntuh:{row.record_id}", "age": np.nan, "male": np.nan,
                         "weight_kg": np.nan, "height_cm": np.nan, "asa": np.nan,
                         "emop": 0, "antype": "General", "department": "NTUH surgery",
                         "duration_min": duration, "or_in_min": 0.0, "an_start_min": 0.0,
                         "surgery_start_min": np.nan, "surgery_end_min": duration,
                         "an_end_min": duration, "or_out_min": duration,
                         "icu_in_min": np.nan, "icu_out_min": np.nan, "death_min": np.nan,
                         "phase_events_available": False})
        add_clean_rows(store, ep, rows)
        clean.append(rows)
        source_audit["selected_surgical_ecg"] += 1
        source_audit["five_minute_measurements"] += len(rows)
        manifest.append({"source_id": row.record_id, "status": "included",
                         "surgery_screen": "Raw ECG records from documented NTUH surgery anesthesia cohort",
                         "prior_ecg_qc": "included", "duration_min": round(duration, 2),
                         "clean_rows": len(rows), **hr_qc})
    pd.concat(clean, ignore_index=True).to_csv(cleaned_path, index=False) if clean else pd.DataFrame().to_csv(cleaned_path, index=False)
    return episodes, manifest, source_audit


def process_source(name: str, parser, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    store_path = out_dir / "validation_store.sqlite"
    db = ValidationStore(store_path, reset=True)
    cleaned_path = out_dir / "cleaned_5min_measurements.csv"
    episodes, manifest, audit = parser(db, cleaned_path)
    db.finish(); db.close()
    if not episodes:
        raise RuntimeError(f"No screen-positive surgical episodes in {name}")
    frame = pd.DataFrame(episodes)
    # Keep each source's ep indexes dense for the shared writer and database.
    remap = {int(old): new for new, old in enumerate(frame.ep_idx)}
    frame["ep_idx"] = frame.ep_idx.map(remap)
    if any(remap[k] != k for k in remap):
        con = sqlite3.connect(store_path)
        for old, new in remap.items():
            if old != new:
                con.execute("UPDATE obs SET ep=? WHERE ep=?", (new, old))
        con.commit(); con.close()
    sequence_dir = OUT / f"maomao_{name}"
    if sequence_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {sequence_dir}")
    source_manifest = manifest
    meta = write_validation_dataset(
        frame, store_path, sequence_dir, TRAIN, name,
        source_manifest, audit,
        add_phase_summaries=bool(audit.get("phase_summaries_available", True)))
    return {"source": name, "clean_manifest": manifest, "source_audit": audit,
            "sequence_meta": meta, "sequence_dir": str(sequence_dir)}


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output.resolve()
    if OUT.exists():
        raise SystemExit(f"Output directory exists; move it first or use a new --output: {OUT}")
    OUT.mkdir(parents=True)
    results = []
    results.append(process_source("uq_vital_signs32_surgical", parse_uq, OUT / "prep_uq"))
    results.append(process_source("auckland_asac_eds25_surgical", parse_asac, OUT / "prep_asac"))
    results.append(process_source("ntuh110_ecg_derived_hr", parse_ntuh, OUT / "prep_ntuh"))
    (OUT / "preprocessing_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    print(json.dumps([{k: v for k, v in x.items() if k != "clean_manifest"} for x in results],
                     ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
