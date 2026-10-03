#!/usr/bin/env python3
"""Convert eICU and SICdb into the frozen MAOMAO v5 external-event contract.

These are ICU-admission timelines, not operation episodes. SICdb offsets can
include pre-ICU surgery; eICU offsets begin at the ICU stay. Both are explicitly
anchored at ICU admission when represented in the resulting metadata.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.external_validation_common import ValidationStore, write_validation_dataset  # noqa: E402


EICU_VITALS = {
    "temperature": "ward_vitals:bt", "sao2": "ward_vitals:spo2",
    "heartrate": "ward_vitals:hr", "respiration": "ward_vitals:rr",
    "systemicsystolic": "vitals:art_sbp", "systemicdiastolic": "vitals:art_dbp",
    "systemicmean": "vitals:art_mbp", "noninvasivesystolic": "vitals:nibp_sbp",
    "noninvasivediastolic": "vitals:nibp_dbp", "noninvasivemean": "vitals:nibp_mbp",
    "cvp": "vitals:cvp", "etco2": "vitals:etco2", "pasystolic": "vitals:pap_sbp",
    "padiastolic": "vitals:pap_dbp", "pamean": "vitals:pap_mbp",
}


def norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def lab_feature(label: object) -> str | None:
    s = norm(label)
    rules = (
        (("creatinine", "kreatinin"), "labs:creatinine"),
        (("lactate", "laktat"), "labs:lactate"),
        (("glucose", "glukose", "bloodsugar"), "labs:glucose"),
        (("hemoglobin", "haemoglobin", "hgb", "hb"), "labs:hb"),
        (("hematocrit", "haematocrit", "hct"), "labs:hct"),
        (("platelet", "thrombozyt"), "labs:platelet"),
        (("potassium", "kalium"), "labs:potassium"),
        (("sodium", "natrium"), "labs:sodium"),
        (("calcium", "kalzium"), "labs:calcium"),
        (("phosphat", "phosphor"), "labs:phosphorus"),
        (("bilirubin", "bilirubin"), "labs:total_bilirubin"),
        (("albumin", "albumin"), "labs:albumin"),
        (("inr", "quick"), "labs:ptinr"), (("aptt", "ptt"), "labs:aptt"),
        (("fibrinogen",), "labs:fibrinogen"),
        (("tropinin", "troponini", "troponin"), "labs:troponin_i"),
        (("so2", "oxygensaturation"), "labs:sao2"),
        (("crp", "creaktivesprotein"), "labs:crp"),
        (("leukocyte", "leukozyt", "wbc"), "labs:wbc"),
        (("aspartateaminotransferase", "got", "ast"), "labs:ast"),
        (("alanineaminotransferase", "gpt", "alt"), "labs:alt"),
        (("urea", "bun", "harnstoff"), "labs:bun"),
        (("bicarbonate", "bikarbonat", "hco3"), "labs:hco3"),
        (("baseexcess", "basedeficit"), "labs:be"),
        (("pco2", "paco2"), "labs:paco2"), (("po2", "pao2"), "labs:pao2"),
        (("ph",), "labs:ph"),
    )
    for terms, feature in rules:
        if any(term in s for term in terms):
            return feature
    return None


def eicu_chart_feature(label: object, *, respiratory: bool) -> str | None:
    """Map timestamped respiratory and nursing chart labels to trained signals."""
    s = norm(label)
    if respiratory:
        if "plateaupressure" in s:
            return "vitals:pplat"
        if "peep" in s or "cpap" in s:
            return "vitals:peep"
        if "peakinsp" in s or "peakpressure" in s or s in {"pip", "pipmmhg"}:
            return "vitals:pip"
        if "fio2" in s or "oxygenpercentage" in s:
            return "vitals:fio2"
        if "exhaledmv" in s or "minuteventil" in s or "minutevolume" in s or s in {"mve", "mvexp"}:
            return "vitals:minvol"
        if "etco2" in s or "endtidalco2" in s:
            return "vitals:etco2"
        if "sao2" in s or "spo2" in s or "oxygensaturation" in s:
            return "ward_vitals:spo2"
        if "respiratoryrate" in s or "rrpatient" in s or "totalrr" in s or s in {"rr", "rrspont"}:
            return "ward_vitals:rr"
        if "setrr" in s or "ventrate" in s:
            return "ward_vitals:rr"
        return None

    if s in {"temperature", "temp"}:
        return "ward_vitals:bt"
    if s in {"heartrate", "pulse"}:
        return "ward_vitals:hr"
    if s in {"respiratoryrate", "resprate"}:
        return "ward_vitals:rr"
    if s in {"oxygensaturation", "spo2", "sao2"}:
        return "ward_vitals:spo2"
    if "arteriallinemap" in s or s == "mapmmhg":
        return "vitals:art_mbp"
    if s in {"cvp", "centralvenouspressure"}:
        return "vitals:cvp"
    if s in {"ci", "cardiacindex"}:
        return "vitals:ci"
    if s in {"endtidalco2", "etco2"}:
        return "vitals:etco2"
    if s in {"bedsideglucose", "bloodglucose"}:
        return "labs:glucose"
    if s in {"bestmotorresponse", "motorresponse"}:
        return "ward_vitals:gcs_m"
    if s in {"besteyeresponse", "eyeopening"}:
        return "ward_vitals:gcs_e"
    return None


def ingest_eicu_charting(store: ValidationStore, root: Path, episodes: pd.DataFrame,
                         surgical_stay_ids: set[int], chunksize: int,
                         *, respiratory: bool) -> Counter:
    table = "respiratoryCharting.csv.gz" if respiratory else "nurseCharting.csv.gz"
    label_col = "respchartvaluelabel" if respiratory else "nursingchartcelltypevallabel"
    value_col = "respchartvalue" if respiratory else "nursingchartvalue"
    offset_col = "respchartoffset" if respiratory else "nursingchartoffset"
    feature_counts = Counter()
    stay = episodes[["ep_idx", "patientunitstayid", "duration_min"]]
    accepted_rows = 0
    for chunk_index, chunk in enumerate(pd.read_csv(
            root / table, usecols=["patientunitstayid", label_col, value_col, offset_col],
            chunksize=chunksize, low_memory=False), 1):
        chunk = chunk[chunk.patientunitstayid.isin(surgical_stay_ids)].copy()
        if chunk.empty:
            continue
        chunk["feature"] = chunk[label_col].map(
            lambda value: eicu_chart_feature(value, respiratory=respiratory))
        chunk = chunk.dropna(subset=["feature"])
        if chunk.empty:
            continue
        raw_value = chunk[value_col].astype("string")
        chunk["value_num"] = pd.to_numeric(raw_value, errors="coerce")
        if not respiratory:
            # Some GCS components are recorded as "4 - spontaneous".
            missing = chunk.value_num.isna()
            if missing.any():
                chunk.loc[missing, "value_num"] = pd.to_numeric(
                    raw_value[missing].str.extract(r"(-?\d+(?:\.\d+)?)", expand=False),
                    errors="coerce")
        chunk["time_min"] = pd.to_numeric(chunk[offset_col], errors="coerce")
        chunk = chunk.dropna(subset=["value_num", "time_min"]).merge(
            stay, on="patientunitstayid", how="inner")
        chunk = chunk[(chunk.time_min >= 0) & (chunk.time_min <= chunk.duration_min)].copy()
        if chunk.empty:
            continue
        fahrenheit = chunk.feature.eq("ward_vitals:bt") & chunk.value_num.gt(60)
        chunk.loc[fahrenheit, "value_num"] = (chunk.loc[fahrenheit, "value_num"] - 32) * 5 / 9
        fraction_fio2 = chunk.feature.eq("vitals:fio2") & chunk.value_num.le(1.5)
        chunk.loc[fraction_fio2, "value_num"] *= 100
        bounds = {
            "ward_vitals:bt": (25, 45), "ward_vitals:hr": (20, 260),
            "ward_vitals:rr": (1, 100), "ward_vitals:spo2": (40, 100),
            "vitals:art_mbp": (10, 250), "vitals:cvp": (-10, 100),
            "vitals:ci": (0, 15), "vitals:etco2": (0, 120),
            "labs:glucose": (5, 1500), "ward_vitals:gcs_m": (1, 6),
            "ward_vitals:gcs_e": (1, 4), "vitals:pplat": (0, 100),
            "vitals:peep": (0, 60), "vitals:pip": (0, 120),
            "vitals:fio2": (20, 100), "vitals:minvol": (0, 100),
        }
        good = np.zeros(len(chunk), dtype=bool)
        for feature, (low, high) in bounds.items():
            selected = chunk.feature.eq(feature)
            good |= selected.to_numpy() & chunk.value_num.between(low, high).to_numpy()
        chunk = chunk[good].copy()
        if chunk.empty:
            continue
        chunk["bin"] = np.ceil(chunk.time_min / 5.0).astype(np.int32)
        grouped = chunk.groupby(["ep_idx", "bin", "feature"], as_index=False).value_num.agg(
            ["sum", "count"]).reset_index()
        store.add_observations(grouped[
            ["ep_idx", "bin", "feature", "sum", "count"]
        ].itertuples(index=False, name=None))
        feature_counts.update(chunk.feature.value_counts().to_dict())
        accepted_rows += len(chunk)
        if chunk_index % 20 == 0:
            store.commit()
            print(json.dumps({"table": table, "chunks": chunk_index,
                              "accepted_rows": accepted_rows}), flush=True)
    store.commit()
    return Counter({f"{table}:{feature}": int(count)
                    for feature, count in feature_counts.items()})


def get_eicu_episodes(root: Path) -> tuple[pd.DataFrame, dict]:
    patients = pd.read_csv(root / "patient.csv.gz", low_memory=False)
    n = len(patients)
    # eICU has no operative log table.  The APACHE admissionDx form does,
    # however, explicitly classify admissions as Operative or Non-operative.
    # Retain only the explicit Operative branch (including its yes/no OR
    # question); do not infer surgery from SICU labels or generic procedure text.
    admission_dx = pd.read_csv(root / "admissionDx.csv.gz", usecols=[
        "patientunitstayid", "admitdxpath"], low_memory=False)
    path = admission_dx.admitdxpath.fillna("").astype("string").str.lower()
    operative = path.str.contains(
        r"admission diagnosis\|all diagnosis\|operative\|diagnosis"
        r"|admission diagnosis\|was the patient admitted from the o\.r\. or went to the o\.r\. within 4 hours of admission\?\|yes",
        regex=True, na=False)
    surgical_ids = set(admission_dx.loc[operative, "patientunitstayid"].astype(int).tolist())
    patients = patients.drop_duplicates("patientunitstayid", keep="first").copy()
    patients = patients[patients.patientunitstayid.isin(surgical_ids)].copy()
    patients["ep_idx"] = np.arange(len(patients), dtype=np.int64)
    age_text = patients.age.astype("string")
    patients["age"] = pd.to_numeric(age_text, errors="coerce")
    patients.loc[age_text.str.contains(r"^\s*>\s*89", na=False), "age"] = 90
    patients["male"] = patients.gender.astype("string").str.lower().eq("male").astype(float)
    patients["weight_kg"] = pd.to_numeric(patients.admissionweight, errors="coerce")
    patients["height_cm"] = pd.to_numeric(patients.admissionheight, errors="coerce")
    duration = pd.to_numeric(patients.unitdischargeoffset, errors="coerce")
    patients["duration_min"] = duration.where(duration > 0, np.nan)
    patients = patients.dropna(subset=["duration_min"]).copy()
    patients["icu_in_min"] = 0.0
    patients["icu_out_min"] = patients.duration_min
    expired = patients.unitdischargestatus.astype("string").str.lower().str.contains("expired|death", na=False)
    patients["death_min"] = np.where(expired, patients.duration_min, np.nan)
    for name in ("or_in_min", "or_out_min", "an_start_min", "an_end_min",
                 "surgery_start_min", "surgery_end_min"):
        if name not in patients:
            patients[name] = np.nan
    patients["subject_id"] = patients.uniquepid.astype("string")
    patients["admission_id"] = patients.patienthealthsystemstayid.fillna(
        patients.patientunitstayid).astype("string")
    patients["operation_id"] = patients.patientunitstayid.astype("string")
    audit = {
        "raw_patient_rows": n, "valid_unique_icu_stays": len(patients),
        "surgical_eligibility": "explicit APACHE admissionDx Operative branch / OR-within-4-hours=Yes",
        "operative_admissiondx_stays": len(surgical_ids),
        "all_admissiondx_rows": int(len(admission_dx)),
        "anchor": "ICU admission (eICU observationoffset/labresultoffset origin)",
        "pre_icu_data": "not available in these stay-relative tables",
        "unit_discharge_status_counts": patients.unitdischargestatus.fillna("missing").value_counts().to_dict(),
    }
    return patients.reset_index(drop=True), audit


def get_sicdb_episodes(root: Path) -> tuple[pd.DataFrame, dict, dict]:
    cases = pd.read_csv(root / "cases.csv.gz", low_memory=False)
    refs = pd.read_csv(root / "d_references.csv.gz", low_memory=False)
    ref_value = refs.set_index("ReferenceGlobalID").ReferenceValue.to_dict()
    raw_cases = len(cases)
    surgery_type = cases.SurgicalAdmissionType.map(ref_value).astype("string")
    cases = cases[surgery_type.isin(["Elective Surgery", "Urgent Surgery"])].copy()
    surgery_type = cases.SurgicalAdmissionType.map(ref_value).astype("string")
    cases["age"] = pd.to_numeric(cases.AgeOnAdmission, errors="coerce")
    cases["male"] = cases.Sex.map(ref_value).astype("string").str.lower().eq("male").astype(float)
    cases["emop"] = surgery_type.map({"Elective Surgery": "elective", "Urgent Surgery": "urgent"})
    cases["weight_kg"] = pd.to_numeric(cases.WeightOnAdmission, errors="coerce")
    cases.loc[cases.weight_kg > 500, "weight_kg"] /= 1000.0
    cases["height_cm"] = pd.to_numeric(cases.HeightOnAdmission, errors="coerce")
    cases["source_icu_offset_min"] = pd.to_numeric(cases.ICUOffset, errors="coerce") / 60.0
    cases["source_stay_min"] = pd.to_numeric(cases.TimeOfStay, errors="coerce") / 60.0
    # Preserve a maximum of seven pre-ICU days and 30 post-ICU days, matching
    # the trained sequence horizon while using the ICU transfer as the anchor.
    cases["source_start_min"] = (cases.source_icu_offset_min - 7 * 24 * 60).clip(lower=0)
    cases["source_end_min"] = np.minimum(cases.source_stay_min,
                                        cases.source_icu_offset_min + 30 * 24 * 60)
    cases = cases[cases.source_start_min.notna() & cases.source_end_min.gt(cases.source_icu_offset_min)].copy()
    cases["ep_idx"] = np.arange(len(cases), dtype=np.int64)
    cases["duration_min"] = cases.source_end_min - cases.source_start_min
    cases["icu_in_min"] = cases.source_icu_offset_min - cases.source_start_min
    cases["icu_out_min"] = cases.duration_min
    death_offset = pd.to_numeric(cases.OffsetOfDeath, errors="coerce") / 60.0
    cases["death_min"] = death_offset - cases.source_start_min
    cases.loc[~cases.death_min.between(0, cases.duration_min), "death_min"] = np.nan
    for name in ("or_in_min", "or_out_min", "an_start_min", "an_end_min",
                 "surgery_start_min", "surgery_end_min"):
        cases[name] = np.nan
    # SICdb documents these heart-surgery offsets relative to ICU admission,
    # while the sequence timeline is shifted to the admission-relative window.
    operation_start = cases.icu_in_min + pd.to_numeric(
        cases.HeartSurgeryBeginOffset, errors="coerce") / 60.0
    operation_end = cases.icu_in_min + pd.to_numeric(
        cases.HeartSurgeryEndOffset, errors="coerce") / 60.0
    cases["surgery_start_min"] = operation_start
    cases["surgery_end_min"] = operation_end
    cases.loc[~cases.surgery_start_min.between(0, cases.duration_min), "surgery_start_min"] = np.nan
    cases.loc[~cases.surgery_end_min.between(0, cases.duration_min), "surgery_end_min"] = np.nan
    cases["subject_id"] = cases.PatientID.astype("string")
    cases["admission_id"] = cases.CaseID.astype("string")
    cases["operation_id"] = cases.CaseID.astype("string")
    ref_map = refs.set_index("ReferenceGlobalID").ReferenceValue.to_dict()
    referring_names = cases.ReferringUnit.map(ref_map).astype("string")
    department_map = {
        "Allgemeinchirurgie": "GS", "Herzchirurgie": "CTS",
        "Orthopädie": "OS", "Urologie": "UR", "Gynäkologie": "OG",
        "Neurochirurgie": "NS", "Kinderchirurgie": "PED",
    }
    cases["department"] = referring_names.map(department_map)
    signal_map = {}
    for row in refs[refs.ReferenceName.eq("SignalFloat")].itertuples(index=False):
        label = str(row.ReferenceValue)
        f = signal_feature(label)
        if f:
            signal_map[int(row.ReferenceGlobalID)] = f
    lab_map = {}
    for row in refs[refs.ReferenceName.eq("Laboratory")].itertuples(index=False):
        f = lab_feature(row.ReferenceValue)
        if f:
            lab_map[int(row.ReferenceGlobalID)] = f
    audit = {
        "raw_case_rows": int(raw_cases),
        "valid_icu_cases": int(len(cases)),
        "surgical_eligibility": "SurgicalAdmissionType is Elective Surgery or Urgent Surgery; No Surgery and Unknown excluded",
        "surgical_admission_type_counts": surgery_type.value_counts().to_dict(),
        "anchor": "first ICU/intermediate-care transfer (ICUOffset; source offset is seconds from primary admission)",
        "heart_surgery_offset_reference": "HeartSurgeryBeginOffset/EndOffset are seconds relative to ICU admission",
        "pre_icu_window_minutes": 10080,
        "post_icu_window_minutes": 43200,
        "sex_reference_values": {str(k): v for k, v in ref_value.items() if k in {735, 736, 737}},
        "hospital_unit_counts": cases.HospitalUnit.map(ref_map).fillna("missing").value_counts().to_dict(),
        "mapped_referring_department_counts": cases.department.fillna("unmapped").value_counts().to_dict(),
        "heart_surgery_timing_available": int(cases.surgery_start_min.notna().sum()),
        "mapped_signal_dataids": len(signal_map), "mapped_laboratory_ids": len(lab_map),
    }
    return cases.reset_index(drop=True), audit, {"refs": refs, "signal_map": signal_map, "lab_map": lab_map}


def signal_feature(label: str) -> str | None:
    s = norm(label)
    rules = (
        (("arterialsystolic", "bloodpressurearterialsystolic"), "vitals:art_sbp"),
        (("arterialdiastolic", "bloodpressurearterialdiastolic"), "vitals:art_dbp"),
        (("arterialmap", "bloodpressurearterialmap"), "vitals:art_mbp"),
        (("nisystolic", "noninvasivesystolic", "bloodpressurenisystolic"), "vitals:nibp_sbp"),
        (("nidiastolic", "bloodpressurenidiastolic"), "vitals:nibp_dbp"),
        (("nimap", "bloodpressurenimap"), "vitals:nibp_mbp"),
        (("heartrate", "heartrateecg", "heartrateecg"), "ward_vitals:hr"),
        (("heartrate", "puls", "pulse"), "ward_vitals:hr"),
        (("spo2", "oxygensaturation", "sao2"), "ward_vitals:spo2"),
        (("resprate", "respiratoryrate", "respirationrate"), "ward_vitals:rr"),
        (("temperature", "temperatur"), "ward_vitals:bt"),
        (("etco2", "endtidalco2"), "vitals:etco2"),
        (("centralvenouspressure", "cvp"), "vitals:cvp"),
        (("papsystolic",), "vitals:pap_sbp"), (("papdiastolic",), "vitals:pap_dbp"),
        (("papmean",), "vitals:pap_mbp"),
        (("peep",), "vitals:peep"), (("fio2",), "vitals:fio2"),
        (("peakinspiratorypressure", "peakairwaypressure"), "vitals:pip"),
        (("crrtbloodflow", "crrtdialysateflow", "crrtwithdrawal",
          "substituteprae", "substitutepost", "crrtcalciumsu"), "ward_vitals:crrt"),
        (("ecmopumpspeed", "ecmobloodflow"), "ward_vitals:ecmo"),
        (("minutevolume", "mvexp"), "vitals:minvol"),
    )
    for terms, feature in rules:
        if any(term in s for term in terms):
            return feature
    if s == "mv":
        return "vitals:minvol"
    return None


def _add_numeric(store, chunk: pd.DataFrame, episodes: pd.DataFrame, id_col: str,
                 offset_col: str, mapping: dict[str, str], *, offset_scale: float = 1.0,
                 start_col: str | None = None, chunksize: int = 500_000) -> int:
    if not mapping:
        return 0
    ep_cols = ["ep_idx", id_col, "source_start_min", "source_end_min"] if start_col else ["ep_idx", id_col, "duration_min"]
    epmap = episodes[ep_cols].drop_duplicates(id_col)
    long = chunk.melt(id_vars=[id_col, offset_col], value_vars=list(mapping),
                      var_name="raw_feature", value_name="value_num")
    long["feature"] = long.raw_feature.map(mapping)
    long["offset_min"] = pd.to_numeric(long[offset_col], errors="coerce") * offset_scale
    long["value_num"] = pd.to_numeric(long.value_num, errors="coerce")
    long = long.dropna(subset=["offset_min", "value_num"])
    long = long[np.isfinite(long.value_num)]
    if long.empty:
        return 0
    long = long.merge(epmap, on=id_col, how="inner")
    if start_col:
        long["time_min"] = long.offset_min - long.source_start_min
        long = long[(long.time_min >= 0) & (long.offset_min <= long.source_end_min)]
    else:
        long["time_min"] = long.offset_min
        long = long[(long.time_min >= 0) & (long.time_min <= long.duration_min)]
    if long.empty:
        return 0
    long["bin"] = np.ceil(long.time_min / 5.0).astype(np.int32)
    grouped = long.groupby(["ep_idx", "bin", "feature"], as_index=False).value_num.agg(["sum", "count"]).reset_index()
    store.add_observations(grouped[["ep_idx", "bin", "feature", "sum", "count"]].itertuples(index=False, name=None))
    return int(len(long))


def preprocess_eicu(root: Path, output: Path, training_dir: Path, stage: Path, chunksize: int) -> dict:
    episodes, audit = get_eicu_episodes(root)
    store = ValidationStore(stage)
    stay = episodes[["ep_idx", "patientunitstayid", "duration_min"]]
    surgical_stay_ids = set(stay.patientunitstayid.astype(int).tolist())
    vital_tables = (
        ("vitalPeriodic.csv.gz", "observationoffset", {k: v for k, v in EICU_VITALS.items() if k in {
            "temperature", "sao2", "heartrate", "respiration", "systemicsystolic", "systemicdiastolic", "systemicmean", "cvp", "etco2", "pasystolic", "padiastolic", "pamean"}}),
        ("vitalAperiodic.csv.gz", "observationoffset", {k: v for k, v in EICU_VITALS.items() if k.startswith("noninvasive")}),
    )
    obs_count = 0
    for name, offset, features in vital_tables:
        for i, chunk in enumerate(pd.read_csv(root / name, usecols=["patientunitstayid", offset, *features],
                                                 chunksize=chunksize), 1):
            chunk = chunk[chunk.patientunitstayid.isin(surgical_stay_ids)]
            if chunk.empty:
                continue
            chunk = chunk.rename(columns={"patientunitstayid": "stay_id"})
            ep = stay.rename(columns={"patientunitstayid": "stay_id"})
            obs_count += _add_numeric(store, chunk, ep, "stay_id", offset, features, offset_scale=1.0)
            if i % 20 == 0:
                store.commit(); print(json.dumps({"table": name, "chunks": i, "observations": obs_count}), flush=True)
        store.commit()

    chart_feature_counts = Counter()
    chart_feature_counts.update(ingest_eicu_charting(
        store, root, episodes, surgical_stay_ids, chunksize, respiratory=True))
    chart_feature_counts.update(ingest_eicu_charting(
        store, root, episodes, surgical_stay_ids, chunksize, respiratory=False))

    # eICU exposes explicit ventilator start/end offsets. These event markers
    # preserve treatment timing without inferring ventilation from ICU labels.
    vent_event_rows = 0
    stay_lookup = stay.rename(columns={"patientunitstayid": "stay_id"})
    for i, chunk in enumerate(pd.read_csv(
            root / "respiratoryCare.csv.gz",
            usecols=["patientunitstayid", "ventstartoffset", "ventendoffset"],
            chunksize=chunksize), 1):
        chunk = chunk[chunk.patientunitstayid.isin(surgical_stay_ids)].merge(
            stay_lookup, left_on="patientunitstayid", right_on="stay_id", how="inner")
        rows = []
        for column, event_name in (("ventstartoffset", "ventilation_start"),
                                   ("ventendoffset", "ventilation_stop")):
            offset = pd.to_numeric(chunk[column], errors="coerce")
            valid = offset.ge(0) & offset.le(chunk.duration_min)
            rows.extend((int(ep), int(math.ceil(float(t) / 5.0)), event_name, 0.0, 0)
                        for ep, t in zip(chunk.loc[valid, "ep_idx"], offset[valid]))
        store.add_events(rows)
        vent_event_rows += len(rows)
        if i % 20 == 0:
            store.commit()
            print(json.dumps({"table": "respiratoryCare.csv.gz", "chunks": i,
                              "ventilation_events": vent_event_rows}), flush=True)
    store.commit()

    lab_names = ("lactate", "glucose", "creatinine", "sodium", "potassium", "calcium", "magnesium",
                 "phosphate", "phosphorus", "hemoglobin", "hematocrit", "platelet", "bilirubin", "albumin",
                 "inr", "pt", "ptt", "aptt", "fibrinogen", "wbc", "white blood cell", "ast", "alt", "bun",
                 "urea", "hco3", "bicarbonate", "base excess", "pao2", "paco2", "ph")
    lab_features = {name: lab_feature(name) for name in lab_names}
    lab_features = {k: v for k, v in lab_features.items() if v}
    for i, chunk in enumerate(pd.read_csv(root / "lab.csv.gz", usecols=["patientunitstayid", "labresultoffset", "labname", "labresult"], chunksize=chunksize), 1):
        chunk = chunk[chunk.patientunitstayid.isin(surgical_stay_ids)]
        if chunk.empty:
            continue
        chunk["feature"] = chunk.labname.map(lambda x: lab_feature(x))
        chunk = chunk.dropna(subset=["feature"])
        chunk["value_num"] = pd.to_numeric(chunk.labresult, errors="coerce")
        chunk["offset_min"] = pd.to_numeric(chunk.labresultoffset, errors="coerce")
        chunk = chunk.dropna(subset=["value_num", "offset_min"]).merge(
            stay, on="patientunitstayid", how="inner")
        chunk = chunk[(chunk.offset_min >= 0) & (chunk.offset_min <= chunk.duration_min)]
        chunk["bin"] = np.ceil(chunk.offset_min / 5.0).astype(np.int32)
        grouped = chunk.groupby(["ep_idx", "bin", "feature"], as_index=False).value_num.agg(["sum", "count"]).reset_index()
        store.add_observations(grouped[["ep_idx", "bin", "feature", "sum", "count"]].itertuples(index=False, name=None))
        if i % 20 == 0:
            store.commit(); print(json.dumps({"table": "lab.csv.gz", "chunks": i}), flush=True)
    store.commit()

    # Medication contexts use documented administration/start offsets and are
    # matched later against the frozen training vocabulary.
    for name, offset_col in (("infusionDrug.csv.gz", "infusionoffset"), ("medication.csv.gz", "drugstartoffset")):
        cols = ["patientunitstayid", offset_col, "drugname"]
        for i, chunk in enumerate(pd.read_csv(root / name, usecols=cols, chunksize=chunksize), 1):
            chunk = chunk[chunk.patientunitstayid.isin(surgical_stay_ids)]
            if chunk.empty:
                continue
            chunk["time"] = pd.to_numeric(chunk[offset_col], errors="coerce")
            chunk = chunk.dropna(subset=["time", "drugname"]).merge(stay, on="patientunitstayid", how="inner")
            chunk = chunk[(chunk.time >= 0) & (chunk.time <= chunk.duration_min)]
            rows = ((int(ep), int(math.ceil(float(t) / 5)), "medication", str(drug))
                    for ep, t, drug in chunk[["ep_idx", "time", "drugname"]].itertuples(index=False, name=None))
            store.add_context(rows)
            if i % 20 == 0:
                store.commit(); print(json.dumps({"table": name, "chunks": i}), flush=True)
        store.commit()
    store.finish(); store.close()
    manifest = [{"name": name, "size_bytes": (root / name).stat().st_size} for name in (
        "patient.csv.gz", "vitalPeriodic.csv.gz", "vitalAperiodic.csv.gz", "lab.csv.gz",
        "respiratoryCharting.csv.gz", "nurseCharting.csv.gz", "respiratoryCare.csv.gz",
        "infusionDrug.csv.gz", "medication.csv.gz")]
    audit.update({"mapped_numeric_observations_from_periodic_tables": obs_count,
                  "mapped_charting_observations_by_source_feature": dict(chart_feature_counts),
                  "ventilation_start_stop_events": vent_event_rows,
                  "icu_unit_type_counts": episodes.unittype.fillna("missing").value_counts().to_dict(),
                  "time_semantics": "ICU-relative minutes; numeric series aggregated in 5-minute bins",
                  "feature_limitations": "No pre-ICU physiology; diagnosis timestamps are not treated as observation times; ICD-9 diagnosis codes are not coerced into the frozen ICD-10 token vocabulary"})
    return write_validation_dataset(episodes, stage, output, training_dir, "eICU",
                                    manifest, audit, add_phase_summaries=False,
                                    training_unit="icu_stay")


def preprocess_sicdb(root: Path, output: Path, training_dir: Path, stage: Path, chunksize: int) -> dict:
    episodes, audit, maps = get_sicdb_episodes(root)
    refs = maps["refs"]
    signal_map = maps["signal_map"]
    lab_map = maps["lab_map"]
    store = ValidationStore(stage)
    training_meta = json.loads((training_dir / "event_sequence_meta.json").read_text())
    diagnosis_vocab = {name.removeprefix("diagnosis:") for name in training_meta["token_vocabulary"]
                       if name.startswith("diagnosis:")}
    diagnosis_rows = []
    for ep, code in episodes[["ep_idx", "ICD10Main"]].itertuples(index=False, name=None):
        match = re.match(r"^\s*([A-Z][0-9]{2})", str(code).upper())
        if match and match.group(1) in diagnosis_vocab:
            diagnosis_rows.append((int(ep), 0, "diagnosis", match.group(1)))
    store.add_context(diagnosis_rows)
    obs_count = 0
    base = episodes[["ep_idx", "CaseID", "source_start_min", "source_end_min"]]
    surgical_case_ids = set(base.CaseID.astype(int).tolist())
    float_path = root / "data_float_h.csv.gz"
    if signal_map:
        for i, chunk in enumerate(pd.read_csv(float_path, usecols=["CaseID", "DataID", "Offset", "Val"], chunksize=chunksize), 1):
            chunk = chunk[chunk.CaseID.isin(surgical_case_ids)]
            chunk = chunk[chunk.DataID.isin(signal_map)]
            if len(chunk):
                chunk["feature"] = chunk.DataID.map(signal_map)
                chunk["value_num"] = pd.to_numeric(chunk.Val, errors="coerce")
                chunk["offset_min"] = pd.to_numeric(chunk.Offset, errors="coerce") / 60.0
                chunk = chunk.dropna(subset=["value_num", "offset_min"]).merge(base, on="CaseID", how="inner")
                chunk["time_min"] = chunk.offset_min - chunk.source_start_min
                chunk = chunk[(chunk.time_min >= 0) & (chunk.offset_min <= chunk.source_end_min)]
                if len(chunk):
                    chunk["bin"] = np.ceil(chunk.time_min / 5.0).astype(np.int32)
                    grouped = chunk.groupby(["ep_idx", "bin", "feature"], as_index=False).value_num.agg(["sum", "count"]).reset_index()
                    store.add_observations(grouped[["ep_idx", "bin", "feature", "sum", "count"]].itertuples(index=False, name=None))
                    obs_count += len(chunk)
            if i % 30 == 0:
                store.commit(); print(json.dumps({"table": "data_float_h.csv.gz", "chunks": i, "observations": obs_count}), flush=True)
        store.commit()
    if lab_map:
        for i, chunk in enumerate(pd.read_csv(root / "laboratory.csv.gz", usecols=["CaseID", "LaboratoryID", "Offset", "LaboratoryValue"], chunksize=chunksize), 1):
            chunk = chunk[chunk.CaseID.isin(surgical_case_ids)]
            chunk = chunk[chunk.LaboratoryID.isin(lab_map)]
            if len(chunk):
                chunk["feature"] = chunk.LaboratoryID.map(lab_map)
                chunk["value_num"] = pd.to_numeric(chunk.LaboratoryValue, errors="coerce")
                chunk["offset_min"] = pd.to_numeric(chunk.Offset, errors="coerce") / 60.0
                chunk = chunk.dropna(subset=["value_num", "offset_min"]).merge(base, on="CaseID", how="inner")
                chunk["time_min"] = chunk.offset_min - chunk.source_start_min
                chunk = chunk[(chunk.time_min >= 0) & (chunk.offset_min <= chunk.source_end_min)]
                if len(chunk):
                    chunk["bin"] = np.ceil(chunk.time_min / 5.0).astype(np.int32)
                    grouped = chunk.groupby(["ep_idx", "bin", "feature"], as_index=False).value_num.agg(["sum", "count"]).reset_index()
                    store.add_observations(grouped[["ep_idx", "bin", "feature", "sum", "count"]].itertuples(index=False, name=None))
            if i % 30 == 0:
                store.commit(); print(json.dumps({"table": "laboratory.csv.gz", "chunks": i}), flush=True)
        store.commit()

    medref = refs[refs.ReferenceName.eq("Drug")].set_index("ReferenceGlobalID").ReferenceValue.to_dict()
    case_map = episodes[["ep_idx", "CaseID", "source_start_min", "source_end_min"]]
    for i, chunk in enumerate(pd.read_csv(root / "medication.csv.gz", usecols=["CaseID", "DrugID", "Offset"], chunksize=chunksize), 1):
        chunk = chunk[chunk.CaseID.isin(surgical_case_ids)]
        chunk["name"] = chunk.DrugID.map(medref)
        chunk["offset_min"] = pd.to_numeric(chunk.Offset, errors="coerce") / 60.0
        chunk = chunk.dropna(subset=["name", "offset_min"]).merge(case_map, on="CaseID", how="inner")
        chunk["time_min"] = chunk.offset_min - chunk.source_start_min
        chunk = chunk[(chunk.time_min >= 0) & (chunk.offset_min <= chunk.source_end_min)]
        rows = ((int(ep), int(math.ceil(float(t) / 5)), "medication", str(name))
                for ep, t, name in chunk[["ep_idx", "time_min", "name"]].itertuples(index=False, name=None))
        store.add_context(rows)
        if i % 30 == 0:
            store.commit(); print(json.dumps({"table": "medication.csv.gz", "chunks": i}), flush=True)
    store.finish(); store.close()
    manifest = [{"name": p.name, "size_bytes": p.stat().st_size} for p in (
        root / "cases.csv.gz", root / "data_float_h.csv.gz", root / "laboratory.csv.gz",
        root / "medication.csv.gz", root / "d_references.csv.gz")]
    audit.update({"mapped_numeric_observations": obs_count,
                  "mapped_icd10_primary_diagnosis_contexts": len(diagnosis_rows),
                  "time_semantics": "seconds from primary admission shifted to a 7-day pre-ICU / 30-day post-ICU window",
                  "physiology_resolution": "hourly averages from data_float_h; minute rawdata stream is not expanded",
                  "feature_limitations": "EEG/BIS, cerebral oximetry, urine/drainage, and unsupported respiratory signals are outside the frozen model feature contract"})
    return write_validation_dataset(episodes, stage, output, training_dir, "SICdb",
                                    manifest, audit, add_phase_summaries=True,
                                    training_unit="icu_stay")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", choices=("eicu", "sicdb"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training_dir", type=Path, default=Path("data/perioperative_event_sequences_v5_full"))
    parser.add_argument("--stage_db", type=Path, required=True)
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    result = (preprocess_eicu(args.input, args.output, args.training_dir, args.stage_db, args.chunksize)
              if args.source == "eicu" else
              preprocess_sicdb(args.input, args.output, args.training_dir, args.stage_db, args.chunksize))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
