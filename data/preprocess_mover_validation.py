#!/usr/bin/env python3
"""Build the MOVER EPIC external-validation event dataset.

Uses EPIC patient information for operation anchors, cleaned flowsheets for
five-minute physiology, and timestamped EPIC laboratory/medication tables.
Untimestamped billing diagnoses and postoperative complication fields are
audited but excluded from the causal event history.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.external_validation_common import (  # noqa: E402
    ValidationStore, normalise_text, to_datetime, write_validation_dataset,
)


def manifest(root: Path) -> list[dict]:
    names = (
        "EPIC_EMR.tar.gz", "Epic_flowsheets_cleaned.tar.gz",
        "EPIC_patient_measurments.tar.gz", "EPIC_MRN_PAT_ID.csv",
    )
    return [{"name": name, "size_bytes": (root / name).stat().st_size} for name in names]


def ensure_extracted(root: Path) -> None:
    extracted = root / "extracted"
    emr_marker = extracted / "EPIC_EMR/EMR/patient_information.csv"
    flow_marker = extracted / "flowsheets_cleaned"
    extracted.mkdir(parents=True, exist_ok=True)
    if not emr_marker.is_file():
        subprocess.run(["tar", "-xzf", str(root / "EPIC_EMR.tar.gz"), "-C", str(extracted)], check=True)
    if not flow_marker.is_dir() or not any(flow_marker.glob("*.csv")):
        subprocess.run(["tar", "-xzf", str(root / "Epic_flowsheets_cleaned.tar.gz"),
                        "-C", str(extracted)], check=True)


def build_episodes(root: Path) -> tuple[pd.DataFrame, dict]:
    path = root / "extracted/EPIC_EMR/EMR/patient_information.csv"
    frame = pd.read_csv(path, low_memory=False)
    for column in ("HOSP_ADMSN_TIME", "HOSP_DISCH_TIME", "IN_OR_DTTM", "OUT_OR_DTTM",
                   "AN_START_DATETIME", "AN_STOP_DATETIME"):
        frame[column] = to_datetime(frame[column])
    before = len(frame)
    frame = frame.dropna(subset=["LOG_ID", "MRN", "IN_OR_DTTM", "OUT_OR_DTTM"])
    frame = frame[(frame.OUT_OR_DTTM > frame.IN_OR_DTTM) &
                  (frame.OUT_OR_DTTM <= frame.IN_OR_DTTM + pd.Timedelta(hours=24))]
    frame = frame.sort_values(["LOG_ID", "IN_OR_DTTM"]).drop_duplicates("LOG_ID", keep="first").copy()
    frame["episode_start"] = pd.concat([
        frame.HOSP_ADMSN_TIME, frame.IN_OR_DTTM - pd.Timedelta(days=7)], axis=1).max(axis=1)
    operation_end = pd.concat([frame.OUT_OR_DTTM, frame.AN_STOP_DATETIME], axis=1).max(axis=1)
    frame["episode_end"] = pd.concat([
        frame.HOSP_DISCH_TIME, operation_end + pd.Timedelta(days=30)], axis=1).min(axis=1)
    frame = frame[
        frame.episode_start.notna() & frame.episode_end.notna() &
        frame.IN_OR_DTTM.ge(frame.episode_start) & frame.OUT_OR_DTTM.le(frame.episode_end)
    ].sort_values(["MRN", "IN_OR_DTTM"]).reset_index(drop=True)
    frame["ep_idx"] = np.arange(len(frame), dtype=np.int64)
    frame["subject_id"] = frame.MRN.astype(str)
    frame["admission_id"] = frame.LOG_ID.astype(str)
    frame["operation_id"] = frame.LOG_ID.astype(str)
    frame["age"] = pd.to_numeric(frame.BIRTH_DATE, errors="coerce")
    frame["male"] = frame.SEX.astype(str).str.lower().eq("male").astype(float)
    # The MOVER EPIC extract stores weight in ounces (Epic's native unit).
    frame["weight_kg"] = pd.to_numeric(frame.WEIGHT, errors="coerce") / 35.27396195
    frame["height_cm"] = np.nan
    frame["duration_min"] = (frame.episode_end - frame.episode_start).dt.total_seconds() / 60
    for source, target in (
        ("IN_OR_DTTM", "or_in_min"), ("OUT_OR_DTTM", "or_out_min"),
        ("AN_START_DATETIME", "an_start_min"), ("AN_STOP_DATETIME", "an_end_min"),
    ):
        frame[target] = (frame[source] - frame.episode_start).dt.total_seconds() / 60
    frame["surgery_start_min"] = np.nan; frame["surgery_end_min"] = np.nan
    icu = frame.ICU_ADMIN_FLAG.astype(str).str.lower().isin(["yes", "y", "1", "true"])
    # MOVER exposes ICU admission as a flag but not a separate timestamp.  OR exit
    # is therefore retained as an explicitly documented postoperative transfer proxy.
    frame["icu_in_min"] = np.where(icu, frame.or_out_min, np.nan)
    frame["icu_out_min"] = np.nan
    expired = frame.DISCH_DISP.astype(str).str.lower().str.contains("expir|death|deceased", regex=True)
    frame["death_min"] = np.where(
        expired, (frame.HOSP_DISCH_TIME - frame.episode_start).dt.total_seconds() / 60, np.nan)
    asa = pd.to_numeric(frame.ASA_RATING_C, errors="coerce")
    frame["asa"] = asa.map(lambda value: f"{float(value):.1f}" if pd.notna(value) else "")
    frame["antype"] = frame.PRIMARY_ANES_TYPE_NM.fillna("").astype(str).str.strip()
    frame["antype"] = frame["antype"].replace({
        "General Anesthesia": "General", "Spinal": "Neuraxial",
        "Spinal/Epidural": "Neuraxial", "Combined Spinal/Epidural": "Neuraxial",
    })
    frame["department"] = ""
    audit = {
        "patient_information_rows": before,
        "valid_unique_operation_episodes": len(frame),
        "duplicate_or_invalid_rows_removed": before - len(frame),
        "weight_conversion": "EPIC ounces / 35.27396195 = kg",
        "icu_transfer_semantics": "ICU_ADMIN_FLAG=true at OR exit; proxy because no ICU timestamp is released",
        "diagnosis_policy": "patient_visit/history/coding omitted because no chart timestamp is released",
        "postoperative_complication_policy": "audited only; untimestamped fields are not inserted into event history",
    }
    return frame, audit


def flow_feature(name: object) -> str | None:
    text = str(name).strip().lower()
    compact = normalise_text(text)
    if not compact:
        return None
    arterial = any(term in text for term in ("arterial", "a-line", "aline", "art "))
    if "systolic" in text or compact in {"sbp", "nibps", "abps"}:
        return "vitals:art_sbp" if arterial else "vitals:nibp_sbp"
    if "diastolic" in text or compact in {"dbp", "nibpd", "abpd"}:
        return "vitals:art_dbp" if arterial else "vitals:nibp_dbp"
    if "mean arterial" in text or compact in {"map", "nibpm", "abpm"} or compact.startswith("mapmmhg"):
        return "vitals:art_mbp" if arterial else "vitals:nibp_mbp"
    if any(term in compact for term in ("spo2", "pulseox", "oxygensaturation")):
        return "ward_vitals:spo2"
    if compact in {"pulse", "heartrate", "hr"} or "heart rate" in text:
        return "ward_vitals:hr"
    if "respiratory rate" in text or compact in {"rr", "resp", "resprate"}:
        return "ward_vitals:rr"
    if "temperature" in text or compact in {"temp", "temperaturec", "temperaturef"}:
        return "ward_vitals:bt"
    if "end tidal co2" in text or "etco2" in compact:
        return "vitals:etco2"
    if "fio2" in compact or "inspired oxygen" in text:
        return "vitals:fio2"
    if "peep" in compact:
        return "vitals:peep"
    if "plateau" in text and "pressure" in text:
        return "vitals:pplat"
    if ("peak" in text and "pressure" in text) or compact.startswith("pip"):
        return "vitals:pip"
    if "minute ventilation" in text or "minute volume" in text or "min volume" in text:
        return "vitals:minvol"
    if compact in {"cvp", "centralvenouspressure"}:
        return "vitals:cvp"
    if "cardiac index" in text:
        return "vitals:ci"
    if "gcs" in compact and "motor" in compact:
        return "ward_vitals:gcs_m"
    if "gcs" in compact and ("eye" in compact or "eyes" in compact):
        return "ward_vitals:gcs_e"
    if "estimated blood loss" in text or compact == "ebl":
        return "vitals:ebl"
    return None


def lab_feature(name: object) -> str | None:
    text = str(name).strip().lower()
    compact = normalise_text(text)
    rules = (
        (("albumin",), "labs:albumin"), (("alkalinephosphatase", "alkphos"), "labs:alp"),
        (("alanineaminotransferase", "alt"), "labs:alt"), (("aspartateaminotransferase", "ast"), "labs:ast"),
        (("partialthromboplastin", "aptt", "ptt"), "labs:aptt"), (("baseexcess", "base deficit"), "labs:be"),
        (("bloodureanitrogen", "ureanitrogen", "bun"), "labs:bun"), (("ionizedcalcium",), "labs:ica"),
        (("calcium",), "labs:calcium"), (("chloride",), "labs:chloride"),
        (("creatinekinasemb", "ckmb"), "labs:ckmb"), (("creatinekinase", "totalck"), "labs:ck"),
        (("creatinine",), "labs:creatinine"), (("creactiveprotein",), "labs:crp"),
        (("ddimer",), "labs:d_dimer"), (("fibrinogen",), "labs:fibrinogen"),
        (("glucose",), "labs:glucose"), (("hemoglobina1c", "hba1c"), "labs:hba1c"),
        (("hemoglobin",), "labs:hb"), (("bicarbonate", "totalco2", "carbondioxide"), "labs:hco3"),
        (("hematocrit",), "labs:hct"), (("lactate", "lacticacid"), "labs:lactate"),
        (("lymphocyte",), "labs:lymphocyte"), (("pco2",), "labs:paco2"), (("po2",), "labs:pao2"),
        (("phosphorus", "phosphate"), "labs:phosphorus"), (("platelet",), "labs:platelet"),
        (("potassium",), "labs:potassium"), (("inr",), "labs:ptinr"),
        (("oxygen saturation", "sao2"), "labs:sao2"), (("sodium",), "labs:sodium"),
        (("bilirubintotal", "totalbilirubin"), "labs:total_bilirubin"),
        (("totalprotein",), "labs:total_protein"), (("troponini",), "labs:troponin_i"),
        (("whitebloodcell", "wbc", "leukocytes"), "labs:wbc"),
    )
    # pH is deliberately last to avoid matching phosphate.
    for terms, feature in rules:
        if any(term in compact or term in text for term in terms):
            return feature
    if compact in {"ph", "arterialph", "venousph"}:
        return "labs:ph"
    return None


def convert_units(frame: pd.DataFrame, source: str) -> None:
    units = frame["units"].fillna("").astype(str).str.lower()
    values = frame["value_num"]
    if source == "flow":
        is_temp = frame.feature.eq("ward_vitals:bt")
        fahrenheit = is_temp & (units.str.contains("f") | values.gt(60))
        frame.loc[fahrenheit, "value_num"] = (values[fahrenheit] - 32) * 5 / 9
        fio2 = frame.feature.eq("vitals:fio2") & values.le(1.5)
        frame.loc[fio2, "value_num"] *= 100
    else:
        mmol_glucose = frame.feature.eq("labs:glucose") & units.str.contains("mmol")
        frame.loc[mmol_glucose, "value_num"] *= 18.0182
        umol_creatinine = frame.feature.eq("labs:creatinine") & units.str.contains("umol|µmol", regex=True)
        frame.loc[umol_creatinine, "value_num"] /= 88.4
        gl_albumin = frame.feature.eq("labs:albumin") & units.str.fullmatch(r"g/l")
        frame.loc[gl_albumin, "value_num"] /= 10
        gl_hb = frame.feature.eq("labs:hb") & units.str.fullmatch(r"g/l")
        frame.loc[gl_hb, "value_num"] /= 10
        umol_bili = frame.feature.eq("labs:total_bilirubin") & units.str.contains("umol|µmol", regex=True)
        frame.loc[umol_bili, "value_num"] /= 17.104


def add_measurements(store, chunk, episode_map, time_col, name_col, value_col, unit_col, mapper, source, audit):
    chunk = chunk[chunk.LOG_ID.isin(set(episode_map.LOG_ID))].copy()
    if chunk.empty:
        return
    chunk[time_col] = to_datetime(chunk[time_col])
    chunk["value_num"] = pd.to_numeric(chunk[value_col], errors="coerce")
    chunk["feature"] = chunk[name_col].map(mapper)
    chunk["units"] = chunk[unit_col] if unit_col in chunk else ""
    chunk = chunk.dropna(subset=[time_col, "value_num", "feature"])
    chunk = chunk.merge(episode_map, on="LOG_ID", how="inner")
    chunk = chunk[chunk[time_col].between(chunk.episode_start, chunk.episode_end)].copy()
    if chunk.empty:
        return
    convert_units(chunk, source)
    chunk = chunk[np.isfinite(chunk.value_num)]
    chunk["bin"] = np.ceil((chunk[time_col] - chunk.episode_start).dt.total_seconds() / 300).astype(int)
    grouped = chunk.groupby(["ep_idx", "bin", "feature"], as_index=False).value_num.agg(["sum", "count"]).reset_index()
    store.add_observations(grouped[["ep_idx", "bin", "feature", "sum", "count"]].itertuples(index=False, name=None))
    audit[f"accepted_{source}_rows"] += len(chunk)


def preprocess(args):
    root = Path(args.input).resolve(); output = Path(args.output).resolve()
    ensure_extracted(root)
    episodes, base_audit = build_episodes(root)
    episode_map = episodes[["ep_idx", "LOG_ID", "episode_start", "episode_end"]]
    stage = Path(args.stage_db).resolve(); stage.parent.mkdir(parents=True, exist_ok=True)
    store = ValidationStore(stage); audit = Counter()

    flow_dir = root / "extracted/flowsheets_cleaned"
    flow_files = sorted(flow_dir.glob("*.csv"))
    if not flow_files:
        raise FileNotFoundError(f"No cleaned MOVER flowsheet CSV files under {flow_dir}")
    for file_index, path in enumerate(flow_files, 1):
        for chunk_index, chunk in enumerate(pd.read_csv(
            path, usecols=["LOG_ID", "FLO_DISPLAY_NAME", "RECORDED_TIME", "MEAS_VALUE", "UNITS"],
            chunksize=args.chunksize, low_memory=False,
        ), 1):
            add_measurements(store, chunk, episode_map, "RECORDED_TIME", "FLO_DISPLAY_NAME",
                             "MEAS_VALUE", "UNITS", flow_feature, "flow", audit)
            if chunk_index % 10 == 0:
                store.commit(); print(json.dumps({"table": path.name, "chunks": chunk_index, **dict(audit)}), flush=True)
        store.commit(); print(json.dumps({"flowsheet_files_complete": file_index, "total": len(flow_files)}), flush=True)

    lab_path = root / "extracted/EPIC_EMR/EMR/patient_labs.csv"
    for i, chunk in enumerate(pd.read_csv(
        lab_path,
        usecols=["LOG_ID", "Lab Name", "Observation Value", "Measurement Units", "Collection Datetime"],
        chunksize=args.chunksize, low_memory=False,
    ), 1):
        chunk = chunk.rename(columns={"Measurement Units": "units"})
        add_measurements(store, chunk, episode_map, "Collection Datetime", "Lab Name",
                         "Observation Value", "units", lab_feature, "lab", audit)
        if i % 10 == 0: store.commit(); print(json.dumps({"table": "patient_labs", "chunks": i, **dict(audit)}), flush=True)
    store.commit()

    med_path = root / "extracted/EPIC_EMR/EMR/patient_medications.csv"
    for i, chunk in enumerate(pd.read_csv(
        med_path,
        usecols=["LOG_ID", "MEDICATION_NM", "DISPLAY_NAME", "MAR_ACTION_NM", "MED_ACTION_TIME"],
        chunksize=args.chunksize, low_memory=False,
    ), 1):
        chunk = chunk[chunk.LOG_ID.isin(set(episode_map.LOG_ID))].copy()
        if chunk.empty:
            continue
        action = chunk.MAR_ACTION_NM.fillna("").astype(str).str.lower()
        chunk = chunk[action.str.contains("given|administer|started|infusing", regex=True)].copy()
        chunk["MED_ACTION_TIME"] = to_datetime(chunk.MED_ACTION_TIME)
        chunk["name"] = chunk.MEDICATION_NM.fillna(chunk.DISPLAY_NAME).fillna("")
        chunk = chunk.dropna(subset=["MED_ACTION_TIME"])
        chunk = chunk.merge(episode_map, on="LOG_ID", how="inner")
        chunk = chunk[chunk.MED_ACTION_TIME.between(chunk.episode_start, chunk.episode_end)].copy()
        if len(chunk):
            chunk["bin"] = np.ceil((chunk.MED_ACTION_TIME - chunk.episode_start).dt.total_seconds() / 300).astype(int)
            context = chunk[chunk.name.ne("")][["ep_idx", "bin", "name"]].copy()
            context.insert(2, "kind", "medication")
            store.add_context(context.itertuples(index=False, name=None))
            audit["accepted_medication_rows"] += len(chunk)
        if i % 10 == 0: store.commit(); print(json.dumps({"table": "patient_medications", "chunks": i, **dict(audit)}), flush=True)
    store.finish(); store.close()

    columns = [
        "ep_idx", "subject_id", "admission_id", "operation_id", "LOG_ID", "episode_start", "episode_end",
        "duration_min", "age", "male", "weight_kg", "height_cm", "or_in_min", "or_out_min",
        "an_start_min", "an_end_min", "surgery_start_min", "surgery_end_min", "icu_in_min",
        "icu_out_min", "death_min", "asa", "antype", "department",
    ]
    meta = write_validation_dataset(
        episodes[columns], stage, output, Path(args.training_dir), "MOVER EPIC",
        manifest(root), {**base_audit, **dict(audit), "cleaned_flowsheet_files": [p.name for p in flow_files]},
    )
    print(json.dumps({"complete": True, "output": str(output), "episodes": len(episodes), "tokens": meta["num_tokens"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/MOVER")
    parser.add_argument("--training_dir", default="data/perioperative_event_sequences_v5_full")
    parser.add_argument("--output", default="data/val_mover_v5")
    parser.add_argument("--stage_db", default="data/val_mover_staging.sqlite")
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    preprocess(args)


if __name__ == "__main__":
    main()
