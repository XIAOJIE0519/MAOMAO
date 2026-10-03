#!/usr/bin/env python3
"""Build a perioperative external-validation set from MIMIC-IV 3.1.

Only ICU stays with a timestamped OR Sent -> OR Received pair are retained.
These two MetaVision events are used as explicit OR transfer proxies; ICD
procedure dates are never promoted to invented minute-level timestamps.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.external_validation_common import (  # noqa: E402
    ValidationStore, to_datetime, write_validation_dataset,
)


OR_SENT = 225470
OR_RECEIVED = 225469

CHART_FEATURES = {
    220045: "ward_vitals:hr",
    220050: "vitals:art_sbp", 220051: "vitals:art_dbp", 220052: "vitals:art_mbp",
    220179: "vitals:nibp_sbp", 220180: "vitals:nibp_dbp", 220181: "vitals:nibp_mbp",
    220210: "ward_vitals:rr", 220277: "ward_vitals:spo2",
    220339: "vitals:peep", 223835: "vitals:fio2",
    224695: "vitals:pip", 224696: "vitals:pplat", 228640: "vitals:etco2",
    223761: "ward_vitals:bt", 223762: "ward_vitals:bt",
}

LAB_FEATURES = {
    50809: "labs:glucose", 50931: "labs:glucose", 52569: "labs:glucose",
    50811: "labs:hb", 51222: "labs:hb", 51640: "labs:hb",
    50813: "labs:lactate", 52442: "labs:lactate", 53154: "labs:lactate",
    50818: "labs:paco2", 50820: "labs:ph", 50821: "labs:pao2",
    50862: "labs:albumin", 53085: "labs:albumin",
    50882: "labs:hco3", 50889: "labs:crp", 50910: "labs:ck",
    50912: "labs:creatinine", 52546: "labs:creatinine",
    50970: "labs:phosphorus", 50971: "labs:potassium", 52610: "labs:potassium",
    50983: "labs:sodium", 52623: "labs:sodium",
    51002: "labs:troponin_i", 52642: "labs:troponin_i",
    51221: "labs:hct", 51638: "labs:hct", 51639: "labs:hct", 52028: "labs:hct",
    51237: "labs:ptinr", 51675: "labs:ptinr",
    51265: "labs:platelet", 53189: "labs:platelet",
    51275: "labs:aptt", 52923: "labs:aptt",
    51301: "labs:wbc", 51755: "labs:wbc", 51756: "labs:wbc",
    51623: "labs:fibrinogen", 52116: "labs:fibrinogen",
}


def source_manifest(root: Path) -> list[dict]:
    names = (
        "hosp/patients.csv.gz", "hosp/admissions.csv.gz", "hosp/labevents.csv.gz",
        "icu/icustays.csv.gz", "icu/procedureevents.csv.gz", "icu/chartevents.csv.gz",
        "icu/inputevents.csv.gz", "icu/outputevents.csv.gz", "icu/d_items.csv.gz",
    )
    return [{"name": name, "size_bytes": (root / name).stat().st_size} for name in names]


def build_episodes(root: Path, chunksize: int) -> tuple[pd.DataFrame, dict]:
    events = []
    for chunk in pd.read_csv(
        root / "icu/procedureevents.csv.gz",
        usecols=["subject_id", "hadm_id", "stay_id", "starttime", "itemid", "patientweight"],
        chunksize=chunksize,
    ):
        use = chunk[chunk.itemid.isin([OR_SENT, OR_RECEIVED])].copy()
        if len(use):
            use["starttime"] = to_datetime(use["starttime"])
            events.append(use.dropna(subset=["starttime"]))
    events = pd.concat(events, ignore_index=True)

    pairs = []
    unmatched_sent = unmatched_received = 0
    for stay_id, group in events.sort_values("starttime").groupby("stay_id", sort=True):
        sent = list(group.loc[group.itemid.eq(OR_SENT), "starttime"])
        received = list(group.loc[group.itemid.eq(OR_RECEIVED), "starttime"])
        used = set()
        identity = group.iloc[0]
        weight = pd.to_numeric(group.patientweight, errors="coerce").dropna()
        for start in sent:
            candidates = [(i, end) for i, end in enumerate(received)
                          if i not in used and start < end <= start + pd.Timedelta(hours=24)]
            if not candidates:
                unmatched_sent += 1
                continue
            index, end = candidates[0]
            used.add(index)
            pairs.append({
                "subject_id": int(identity.subject_id), "hadm_id": int(identity.hadm_id),
                "stay_id": int(stay_id), "or_in": start, "or_out": end,
                "weight_kg": float(weight.median()) if len(weight) else np.nan,
            })
        unmatched_received += len(received) - len(used)
    pairs = pd.DataFrame(pairs)
    if pairs.empty:
        raise RuntimeError("No timestamped MIMIC OR Sent/OR Received pairs found")

    admissions = pd.read_csv(root / "hosp/admissions.csv.gz", usecols=[
        "subject_id", "hadm_id", "admittime", "dischtime", "deathtime"])
    for column in ("admittime", "dischtime", "deathtime"):
        admissions[column] = to_datetime(admissions[column])
    patients = pd.read_csv(root / "hosp/patients.csv.gz", usecols=[
        "subject_id", "gender", "anchor_age", "anchor_year"])
    stays = pd.read_csv(root / "icu/icustays.csv.gz", usecols=["stay_id", "intime", "outtime"])
    stays["intime"] = to_datetime(stays["intime"]); stays["outtime"] = to_datetime(stays["outtime"])

    episodes = pairs.merge(admissions, on=["subject_id", "hadm_id"], validate="many_to_one")
    episodes = episodes.merge(patients, on="subject_id", validate="many_to_one")
    episodes = episodes.merge(stays, on="stay_id", validate="many_to_one")
    episodes["episode_start"] = pd.concat([
        episodes.admittime, episodes.or_in - pd.Timedelta(days=7)], axis=1).max(axis=1)
    terminal = pd.concat([episodes.dischtime, episodes.deathtime], axis=1).min(axis=1)
    episodes["episode_end"] = pd.concat([
        terminal, episodes.or_out + pd.Timedelta(days=30)], axis=1).min(axis=1)
    episodes = episodes[
        episodes.episode_start.notna() & episodes.episode_end.notna() &
        episodes.or_in.ge(episodes.episode_start) & episodes.or_out.le(episodes.episode_end)
    ].copy().sort_values(["subject_id", "hadm_id", "or_in"]).reset_index(drop=True)
    episodes["ep_idx"] = np.arange(len(episodes), dtype=np.int64)
    episodes["operation_id"] = episodes.stay_id.astype(str) + ":" + episodes.groupby("stay_id").cumcount().astype(str)
    episodes["admission_id"] = episodes.hadm_id.astype(str)
    episodes["age"] = episodes.anchor_age + (episodes.or_in.dt.year - episodes.anchor_year)
    episodes["male"] = episodes.gender.eq("M").astype(float)
    episodes["height_cm"] = np.nan
    episodes["duration_min"] = (episodes.episode_end - episodes.episode_start).dt.total_seconds() / 60
    episodes["or_in_min"] = (episodes.or_in - episodes.episode_start).dt.total_seconds() / 60
    episodes["or_out_min"] = (episodes.or_out - episodes.episode_start).dt.total_seconds() / 60
    episodes["an_start_min"] = np.nan; episodes["an_end_min"] = np.nan
    episodes["surgery_start_min"] = np.nan; episodes["surgery_end_min"] = np.nan
    episodes["icu_in_min"] = np.nan
    # OR Received means the patient was received back into the ICU.
    episodes["icu_in_min"] = episodes["or_out_min"]
    episodes["icu_out_min"] = np.where(
        episodes.outtime.between(episodes.episode_start, episodes.episode_end),
        (episodes.outtime - episodes.episode_start).dt.total_seconds() / 60, np.nan)
    episodes["death_min"] = np.where(
        episodes.deathtime.between(episodes.episode_start, episodes.episode_end),
        (episodes.deathtime - episodes.episode_start).dt.total_seconds() / 60, np.nan)
    episodes["asa"] = np.nan; episodes["antype"] = ""; episodes["department"] = ""
    audit = {
        "or_sent_rows": int(events.itemid.eq(OR_SENT).sum()),
        "or_received_rows": int(events.itemid.eq(OR_RECEIVED).sum()),
        "paired_operation_episodes": len(episodes),
        "unmatched_or_sent": unmatched_sent,
        "unmatched_or_received": unmatched_received,
        "anchor_semantics": "MIMIC MetaVision OR Sent -> OR Received proxy within one ICU stay",
        "diagnosis_policy": "omitted because diagnoses_icd has no chart timestamp",
    }
    return episodes, audit


def add_numeric_chunk(store, chunk, episode_map, key, time_col, feature_map, audit, conversion=None):
    chunk = chunk[chunk.itemid.isin(feature_map)].copy()
    if chunk.empty:
        return
    chunk[time_col] = to_datetime(chunk[time_col])
    chunk["value_num"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
    chunk = chunk.dropna(subset=[time_col, "value_num"])
    chunk = chunk.merge(episode_map, on=key, how="inner")
    chunk = chunk[chunk[time_col].between(chunk.episode_start, chunk.episode_end)].copy()
    if chunk.empty:
        return
    chunk["feature"] = chunk.itemid.map(feature_map)
    if conversion:
        conversion(chunk)
    chunk = chunk[np.isfinite(chunk.value_num)]
    chunk["bin"] = np.ceil((chunk[time_col] - chunk.episode_start).dt.total_seconds() / 300).astype(int)
    grouped = chunk.groupby(["ep_idx", "bin", "feature"], as_index=False).value_num.agg(["sum", "count"]).reset_index()
    store.add_observations(grouped[["ep_idx", "bin", "feature", "sum", "count"]].itertuples(index=False, name=None))
    audit["accepted_numeric_rows"] += len(chunk)


def preprocess(args):
    root = Path(args.input).resolve(); output = Path(args.output).resolve()
    episodes, base_audit = build_episodes(root, args.chunksize)
    stage = Path(args.stage_db).resolve()
    stage.parent.mkdir(parents=True, exist_ok=True)
    store = ValidationStore(stage)
    audit = Counter()

    stay_map = episodes[["ep_idx", "stay_id", "episode_start", "episode_end"]]
    hadm_map = episodes[["ep_idx", "hadm_id", "episode_start", "episode_end"]]

    def convert_chart(frame):
        fahrenheit = frame.itemid.eq(223761)
        frame.loc[fahrenheit, "value_num"] = (frame.loc[fahrenheit, "value_num"] - 32) * 5 / 9
        fio2 = frame.itemid.eq(223835) & frame.value_num.le(1.5)
        frame.loc[fio2, "value_num"] *= 100

    for i, chunk in enumerate(pd.read_csv(
        root / "icu/chartevents.csv.gz",
        usecols=["stay_id", "charttime", "itemid", "valuenum"], chunksize=args.chunksize,
    ), 1):
        add_numeric_chunk(store, chunk, stay_map, "stay_id", "charttime", CHART_FEATURES, audit, convert_chart)
        if i % 20 == 0:
            store.commit(); print(json.dumps({"table": "chartevents", "chunks": i, **dict(audit)}), flush=True)
    store.commit()

    for i, chunk in enumerate(pd.read_csv(
        root / "hosp/labevents.csv.gz",
        usecols=["hadm_id", "charttime", "itemid", "valuenum"], chunksize=args.chunksize,
    ), 1):
        chunk = chunk.dropna(subset=["hadm_id"]); chunk["hadm_id"] = chunk.hadm_id.astype("int64")
        add_numeric_chunk(store, chunk, hadm_map, "hadm_id", "charttime", LAB_FEATURES, audit)
        if i % 20 == 0:
            store.commit(); print(json.dumps({"table": "labevents", "chunks": i, **dict(audit)}), flush=True)
    store.commit()

    items = pd.read_csv(root / "icu/d_items.csv.gz", usecols=["itemid", "label"]).set_index("itemid").label.to_dict()
    for i, chunk in enumerate(pd.read_csv(
        root / "icu/inputevents.csv.gz",
        usecols=["stay_id", "starttime", "itemid", "amount", "rate", "statusdescription"],
        chunksize=args.chunksize,
    ), 1):
        chunk = chunk[chunk.stay_id.isin(set(stay_map.stay_id))].copy()
        if chunk.empty:
            continue
        chunk["starttime"] = to_datetime(chunk.starttime)
        chunk = chunk.merge(stay_map, on="stay_id", how="inner")
        chunk = chunk[chunk.starttime.between(chunk.episode_start, chunk.episode_end)].copy()
        chunk["name"] = chunk.itemid.map(items).fillna("")
        chunk["bin"] = np.ceil((chunk.starttime - chunk.episode_start).dt.total_seconds() / 300).astype(int)
        context = chunk[chunk.name.ne("")][["ep_idx", "bin", "name"]].copy()
        context.insert(2, "kind", "medication")
        store.add_context(context.itertuples(index=False, name=None))
        obs = []
        for row in chunk.itertuples(index=False):
            lower = row.name.lower()
            value = pd.to_numeric(row.rate, errors="coerce")
            if not np.isfinite(value):
                value = pd.to_numeric(row.amount, errors="coerce")
            if not np.isfinite(value):
                value = 1.0
            feature = None
            if "norepinephrine" in lower: feature = "vitals:nepi"
            elif "epinephrine" in lower: feature = "vitals:epi"
            elif "vasopressin" in lower: feature = "vitals:vaso"
            elif "phenylephrine" in lower: feature = "vitals:phe"
            elif "ephedrine" in lower: feature = "vitals:eph"
            elif "dopamine" in lower: feature = "vitals:dopai"
            elif "dobutamine" in lower: feature = "vitals:dobui"
            elif "packed red" in lower or "or packed rbc" in lower: feature = "vitals:rbc"
            elif "fresh frozen plasma" in lower or "or ffp" in lower: feature = "vitals:ffp"
            elif "platelet" in lower: feature = "vitals:pheresis"
            elif "cryoprecipitate" in lower: feature = "vitals:cryo"
            if feature:
                obs.append((int(row.ep_idx), int(row.bin), feature, float(value), 1))
        store.add_observations(obs); audit["accepted_input_rows"] += len(chunk)
        if i % 10 == 0: store.commit()
    store.commit()

    for chunk in pd.read_csv(
        root / "icu/outputevents.csv.gz",
        usecols=["stay_id", "charttime", "itemid", "value"], chunksize=args.chunksize,
    ):
        chunk = chunk[chunk.itemid.eq(226626)].rename(columns={"value": "valuenum"})
        add_numeric_chunk(store, chunk, stay_map, "stay_id", "charttime", {226626: "vitals:ebl"}, audit)
    store.finish(); store.close()

    columns = [
        "ep_idx", "subject_id", "admission_id", "operation_id", "stay_id", "episode_start", "episode_end",
        "duration_min", "age", "male", "weight_kg", "height_cm", "or_in_min", "or_out_min",
        "an_start_min", "an_end_min", "surgery_start_min", "surgery_end_min", "icu_in_min",
        "icu_out_min", "death_min", "asa", "antype", "department",
    ]
    meta = write_validation_dataset(
        episodes[columns], stage, output, Path(args.training_dir), "MIMIC-IV 3.1 surgical ICU OR-transfer cohort",
        source_manifest(root), {**base_audit, **dict(audit)},
    )
    print(json.dumps({"complete": True, "output": str(output), "episodes": len(episodes), "tokens": meta["num_tokens"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/mimic-iv-3.1")
    parser.add_argument("--training_dir", default="data/perioperative_event_sequences_v5_full")
    parser.add_argument("--output", default="data/val_mimic_v5")
    parser.add_argument("--stage_db", default="data/val_mimic_staging.sqlite")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()
    preprocess(args)


if __name__ == "__main__":
    main()
