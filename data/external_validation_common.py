#!/usr/bin/env python3
"""Shared external-validation event-sequence writer.

External source adapters write harmonised five-minute observations and
timestamped context into a small SQLite contract.  This module applies the
same clinical transition ontology used for the INSPIRE training data and
writes the exact EventSequenceDataset binary layout, while freezing the
training vocabulary and outcome mapping.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from maomao.data.clinical_events import extract_measurement_events
from maomao.data.event_extraction import binary_transitions, merge_series, observed_vasopressor_response
from maomao.data.event_sequence import TOKEN_KINDS
from maomao.data.static_features import STATIC_FEATURES, build_static_features
from scripts.preprocess_event_sequences import (
    SUMMARY_FEATURES, _append_baseline_and_rolling_summaries,
)


VASOPRESSOR_TERMS = (
    "norepinephrine", "noradrenaline", "epinephrine", "adrenaline",
    "noradrenalin", "adrenalin",
    "phenylephrine", "vasopressin", "ephedrine", "metaraminol",
    "dopamine", "dobutamine",
)

# External medication tables commonly append salts, strength, route, package
# and brand names to the generic ingredient.  The training vocabulary stores
# the generic ingredient only.  These aliases are deliberately explicit; we
# do not use edit distance because a plausible-looking drug mismatch is more
# harmful than leaving an unsupported drug unmapped.
MEDICATION_ALIASES = {
    "noradrenalin": "norepinephrine",
    "adrenalin": "epinephrine",
    "phenylephrin": "phenylephrine",
    "dobutamin": "dobutamine",
    "albuterol": "salbutamol",
    "enoxaparin": "enoxaparine",
    "nacl": "sodium chloride",
    "kcl": "potassium chloride",
    "humalog": "insulin lispro",
    "dilantin": "phenytoin",
    "bactrim": "sulfamethoxazole",
    "precedex": "dexmedetomidine",
    "ativan": "lorazepam",
    "lasix": "furosemide",
    "keppra": "levetiracetam",
    "protonix": "pantoprazole",
    "dilaudid": "hydromorphone",
    "zosyn": "piperacillin",
    "versed": "midazolam",
    "pepcid": "famotidine",
}

ANESTHESIA_TYPE_ALIASES = {
    "monitoredanesthesiacaremac": "MAC",
    "moderatesedationbynonanesthesiastaffonly": "MAC",
    "epidural": "Neuraxial",
    "spinal": "Neuraxial",
    "spinalepidural": "Neuraxial",
    "local": "Regional",
    "topical": "Regional",
}


def normalise_text(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


class MedicationResolver:
    """Map source medication labels onto the frozen generic-name vocabulary."""

    def __init__(self, token_vocabulary: dict[str, int]):
        self.exact = {
            normalise_text(name.removeprefix("med:")): name
            for name in token_vocabulary if name.startswith("med:")
        }
        # Very short medication strings (for example ``ol``) can be valid in a
        # local source but are unsafe prefix matchers across institutions.
        self.prefixes = sorted(
            (key for key in self.exact if len(key) >= 5),
            key=lambda key: (-len(key), key),
        )
        self.aliases = {}
        for external_name, generic_name in MEDICATION_ALIASES.items():
            target_key = normalise_text(generic_name)
            token = self.exact.get(target_key)
            if token is not None:
                self.aliases[normalise_text(external_name)] = token

    def resolve(self, source_name: object) -> tuple[str | None, str]:
        key = normalise_text(source_name)
        token = self.exact.get(key)
        if token is not None:
            return token, "exact"
        # Longest generic name wins, so ``insulinlispro`` is chosen before the
        # broader ``insulin`` and norepinephrine is never reduced to epinephrine.
        for generic_key in self.prefixes:
            if key.startswith(generic_key):
                return self.exact[generic_key], "generic_prefix"
        for alias, alias_token in self.aliases.items():
            if key.startswith(alias):
                return alias_token, "explicit_alias"
        return None, "unmatched"


def to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", format="mixed")


def minutes_between(values: pd.Series, origin: pd.Series) -> np.ndarray:
    return ((values - origin).dt.total_seconds() / 60.0).to_numpy(float)


class ValidationStore:
    """Disk-backed, duplicate-safe staging store for harmonised records."""

    def __init__(self, path: Path, reset: bool = True):
        self.path = Path(path)
        if reset and self.path.exists():
            self.path.unlink()
        self.db = sqlite3.connect(self.path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS obs(
              ep INTEGER NOT NULL, bin INTEGER NOT NULL, feature TEXT NOT NULL,
              value_sum REAL NOT NULL, value_count INTEGER NOT NULL,
              PRIMARY KEY(ep, bin, feature)) WITHOUT ROWID
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS context(
              ep INTEGER NOT NULL, bin INTEGER NOT NULL, kind TEXT NOT NULL,
              name TEXT NOT NULL,
              PRIMARY KEY(ep, bin, kind, name)) WITHOUT ROWID
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS direct_event(
              ep INTEGER NOT NULL, bin INTEGER NOT NULL, name TEXT NOT NULL,
              value REAL NOT NULL DEFAULT 0, has_value INTEGER NOT NULL DEFAULT 0,
              PRIMARY KEY(ep, bin, name)) WITHOUT ROWID
        """)
        self.db.commit()

    def add_observations(self, rows: Iterable[tuple]) -> None:
        # sqlite3 binds NumPy integer scalars as binary blobs on some builds.
        # Explicit Python scalar conversion is required because later conflict
        # updates add value_count numerically before dividing the sums.
        rows = [(int(ep), int(bin_value), str(feature), float(value_sum), int(value_count))
                for ep, bin_value, feature, value_sum, value_count in rows]
        if not rows:
            return
        self.db.executemany("""
            INSERT INTO obs(ep,bin,feature,value_sum,value_count) VALUES(?,?,?,?,?)
            ON CONFLICT(ep,bin,feature) DO UPDATE SET
              value_sum=obs.value_sum+excluded.value_sum,
              value_count=obs.value_count+excluded.value_count
        """, rows)

    def add_context(self, rows: Iterable[tuple]) -> None:
        rows = list(rows)
        if rows:
            self.db.executemany(
                "INSERT OR IGNORE INTO context(ep,bin,kind,name) VALUES(?,?,?,?)", rows)

    def add_events(self, rows: Iterable[tuple]) -> None:
        rows = list(rows)
        if rows:
            self.db.executemany("""
                INSERT INTO direct_event(ep,bin,name,value,has_value) VALUES(?,?,?,?,?)
                ON CONFLICT(ep,bin,name) DO UPDATE SET
                  value=MAX(direct_event.value,excluded.value),
                  has_value=MAX(direct_event.has_value,excluded.has_value)
            """, rows)

    def commit(self) -> None:
        self.db.commit()

    def finish(self) -> None:
        self.db.commit()
        self.db.execute("CREATE INDEX IF NOT EXISTS context_ep_idx ON context(ep)")
        self.db.execute("CREATE INDEX IF NOT EXISTS event_ep_idx ON direct_event(ep)")
        self.db.commit()

    def close(self) -> None:
        self.db.close()


@dataclass(frozen=True)
class Record:
    time_min: float
    token: str
    kind: int
    outcome: int = -1
    value: float = 0.0
    has_value: int = 0


class BinaryWriter:
    def __init__(self, path: Path, dtype: str):
        self.handle = path.open("wb")
        self.dtype = np.dtype(dtype)
        self.count = 0

    def write(self, values: Iterable) -> None:
        array = np.asarray(list(values), dtype=self.dtype)
        array.tofile(self.handle)
        self.count += len(array)

    def close(self) -> None:
        self.handle.flush()
        self.handle.close()


def _series(features, *names):
    return merge_series([features[name] for name in names if name in features])


def _bp_abnormality(sbp, mbp):
    times, severity = [], []
    if len(sbp[0]):
        times.append(sbp[0]); severity.append(np.maximum(0.0, (90.0 - sbp[1]) / 20.0))
    if len(mbp[0]):
        times.append(mbp[0]); severity.append(np.maximum(0.0, (65.0 - mbp[1]) / 10.0))
    if not times:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    t, s = np.concatenate(times), np.concatenate(severity)
    order = np.argsort(t, kind="stable")
    t, s = t[order], s[order]
    unique, starts = np.unique(t, return_index=True)
    return unique.astype(np.float32), np.maximum.reduceat(s, starts).astype(np.float32)


def _dedupe(records: list[Record]) -> list[Record]:
    unique = {}
    for record in records:
        key = (round(float(record.time_min), 4), record.token, record.outcome)
        previous = unique.get(key)
        if previous is None or record.value > previous.value:
            unique[key] = record
    return sorted(unique.values(), key=lambda row: (row.time_min, row.kind, row.token))


def _insert_clock_tokens(records: list[Record], interval_min: float,
                         max_per_gap: int, measurement_times: np.ndarray) -> list[Record]:
    if "clock" not in TOKEN_KINDS or interval_min <= 0:
        return records
    measurement_times = np.sort(np.asarray(measurement_times, dtype=np.float32))
    result = []
    for previous, current in zip(records, records[1:]):
        result.append(previous)
        count = min(max_per_gap, max(
            0, int(math.ceil((current.time_min - previous.time_min) / interval_min)) - 1))
        for step in range(1, count + 1):
            clock_time = previous.time_min + step * interval_min
            if clock_time >= current.time_min:
                break
            right = int(np.searchsorted(measurement_times, clock_time, side="right"))
            left = int(np.searchsorted(
                measurement_times, clock_time - interval_min, side="left"))
            result.append(Record(
                clock_time, "<CLOCK>", TOKEN_KINDS["clock"],
                value=float(right - left), has_value=1))
    result.append(records[-1])
    return sorted(result, key=lambda row: (row.time_min, row.kind, row.token))


def _insert_phase_summary_tokens(records: list[Record]) -> list[Record]:
    phases = [(0.0, "phase_summary:preop")]
    seen = {phases[0][1]}
    transition_to_phase = {
        "or_entry": "phase_summary:induction", "anesthesia_start": "phase_summary:induction",
        "surgery_start": "phase_summary:surgery", "surgery_end": "phase_summary:emergence",
        "anesthesia_end": "phase_summary:emergence", "or_exit": "phase_summary:pacu",
        "icu_transfer": "phase_summary:icu",
    }
    for record in records:
        if record.outcome >= 0:
            phase = transition_to_phase.get(record.token.removeprefix("event:"))
            if phase and phase not in seen:
                phases.append((record.time_min, phase)); seen.add(phase)
    summaries = [Record(time, phase, TOKEN_KINDS["procedure_context"])
                 for time, phase in phases]
    return _dedupe(records + summaries)


def write_validation_dataset(
    episodes: pd.DataFrame,
    store_path: Path,
    output_dir: Path,
    training_dir: Path,
    source_name: str,
    source_manifest: list[dict],
    source_audit: dict,
    *,
    add_phase_summaries: bool = True,
    training_unit: str = "operation_episode",
) -> dict:
    """Write one external validation set under the frozen training contract."""
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output_dir}")

    train_meta = json.loads((Path(training_dir) / "event_sequence_meta.json").read_text())
    token_vocab = train_meta["token_vocabulary"]
    stored_outcomes = train_meta.get("stored_outcome_vocabulary", train_meta["outcome_vocabulary"])
    outcome_index = {name: index for index, name in enumerate(stored_outcomes)}
    clock_policy = train_meta.get("clock_token_policy", {})
    clock_interval = float(clock_policy.get("interval_minutes", 0.0))
    max_clock_per_gap = int(clock_policy.get("maximum_tokens_per_inter_record_gap", 0))

    medication_resolver = MedicationResolver(token_vocab)
    diagnosis_tokens = {
        normalise_text(name.removeprefix("diagnosis:")): name
        for name in token_vocab if name.startswith("diagnosis:")
    }

    writers = {
        "token_id": BinaryWriter(output_dir / "token_id.bin", "int32"),
        "time_min": BinaryWriter(output_dir / "time_min.bin", "float32"),
        "value": BinaryWriter(output_dir / "value.bin", "float32"),
        "has_value": BinaryWriter(output_dir / "has_value.bin", "uint8"),
        "token_kind": BinaryWriter(output_dir / "token_kind.bin", "uint8"),
        "outcome_class": BinaryWriter(output_dir / "outcome_class.bin", "int16"),
    }
    # The full v5 checkpoint was trained with age and sex only.  A newer
    # version of static_features.py adds five fields, so select the frozen
    # checkpoint's actual static contract instead of silently changing the
    # input width during external preprocessing.
    if int(train_meta["num_static"]) == 2:
        age = pd.to_numeric(episodes.get("age"), errors="coerce").fillna(0.0)
        male = pd.to_numeric(episodes.get("male"), errors="coerce").fillna(0.0)
        static = np.column_stack((age.to_numpy() / 100.0,
                                  male.clip(0, 1).to_numpy())).astype(np.float32)
    else:
        static = build_static_features(episodes)
    np.save(output_dir / "static_baseline.npy", static)

    db = sqlite3.connect(store_path)
    ptr = [0]
    total = 0
    outcome_counts = Counter()
    context_counts = Counter()
    dropped_context = Counter()
    medication_mapping = Counter()
    unmatched_medication_names = Counter()
    response_audit = Counter()

    def outcome(name: str, t: float, value: float = 0.0, has_value: int = 0):
        if name not in outcome_index:
            return None
        token = f"event:{name}"
        if token not in token_vocab:
            return None
        return Record(t, token, TOKEN_KINDS["clinical_outcome"], outcome_index[name], value, has_value)

    try:
        for position, episode in enumerate(episodes.itertuples(index=False)):
            ep = int(episode.ep_idx)
            duration = float(episode.duration_min)
            records: list[Record] = [Record(0.0, "<BOS>", TOKEN_KINDS["boundary"])]

            weight = float(episode.weight_kg) if pd.notna(episode.weight_kg) else math.nan
            height = float(episode.height_cm) if pd.notna(episode.height_cm) else math.nan
            raw_or_in = float(episode.or_in_min)
            known_at = max(0.0, raw_or_in) if math.isfinite(raw_or_in) else 0.0
            if math.isfinite(weight):
                records.append(Record(known_at, "context:weight_kg", TOKEN_KINDS["static_observation"], value=weight, has_value=1))
            if math.isfinite(height):
                records.append(Record(known_at, "context:height_cm", TOKEN_KINDS["static_observation"], value=height, has_value=1))

            phase_columns = (
                ("or_in_min", "or_entry"), ("an_start_min", "anesthesia_start"),
                ("surgery_start_min", "surgery_start"), ("surgery_end_min", "surgery_end"),
                ("an_end_min", "anesthesia_end"), ("or_out_min", "or_exit"),
                ("icu_in_min", "icu_transfer"), ("icu_out_min", "icu_discharge"),
                ("death_min", "inhospital_death"),
            )
            if bool(getattr(episode, "phase_events_available", True)):
                for column, event_name in phase_columns:
                    value = getattr(episode, column, math.nan)
                    if pd.notna(value) and 0 <= float(value) <= duration:
                        rec = outcome(event_name, float(value))
                        if rec:
                            records.append(rec)

            for field, prefix in (("asa", "context:asa:"), ("antype", "context:antype:"), ("department", "context:department:")):
                value = getattr(episode, field, None)
                if pd.notna(value) and str(value).strip():
                    if field == "antype":
                        value = ANESTHESIA_TYPE_ALIASES.get(
                            normalise_text(value), str(value).strip())
                    candidates = [prefix + str(value).strip(), prefix + str(value).strip().lower(), prefix + str(value).strip().title()]
                    token = next((candidate for candidate in candidates if candidate in token_vocab), None)
                    if token:
                        records.append(Record(known_at, token, TOKEN_KINDS["procedure_context"]))
                        context_counts["procedure_context"] += 1
                    else:
                        dropped_context[f"unseen_{field}"] += 1

            feature_arrays = {}
            rows = db.execute(
                "SELECT bin,feature,value_sum/value_count FROM obs WHERE ep=? ORDER BY bin,feature", (ep,)
            ).fetchall()
            grouped = defaultdict(lambda: ([], []))
            for bin_value, feature, value in rows:
                grouped[feature][0].append(float(bin_value) * 5.0)
                grouped[feature][1].append(float(value))
            for feature, (times, values) in grouped.items():
                feature_arrays[feature] = (np.asarray(times, np.float32), np.asarray(values, np.float32))

            if any(token.startswith("summary:") for token in token_vocab):
                _append_baseline_and_rolling_summaries(
                    records, feature_arrays, baseline_time=known_at,
                    operation_start=max(known_at, raw_or_in) if math.isfinite(raw_or_in) else known_at,
                    operation_end=min(float(episode.or_out_min), duration)
                    if math.isfinite(float(episode.or_out_min)) else duration,
                )

            for transition in extract_measurement_events(feature_arrays):
                rec = outcome(transition.name, transition.time_min, transition.severity or transition.value, 1)
                if rec:
                    records.append(rec)

            for feature, start_name, stop_name, gap in (
                ("ward_vitals:vent", "ventilation_start", "ventilation_stop", 360.0),
                ("ward_vitals:crrt", "crrt_start", "crrt_stop", 720.0),
                ("ward_vitals:ecmo", "ecmo_start", "ecmo_stop", 720.0),
                ("ward_vitals:iabp", "iabp_start", "iabp_stop", 720.0),
            ):
                if feature in feature_arrays:
                    for transition in binary_transitions(*feature_arrays[feature], start_name, stop_name, inactivity_gap=gap):
                        rec = outcome(transition.name, transition.time_min, transition.value, 1)
                        if rec:
                            records.append(rec)

            if "vitals:ebl" in feature_arrays:
                major = severe = False
                for t, value in zip(*feature_arrays["vitals:ebl"]):
                    name = None
                    if value >= 500 and not severe:
                        name, severe = "severe_bleeding_signal", True
                    elif value >= 300 and not major:
                        name, major = "major_bleeding_signal", True
                    if name:
                        rec = outcome(name, float(t), float(value), 1)
                        if rec:
                            records.append(rec)

            for feature, event_name in (
                ("vitals:rbc", "rbc_transfusion"), ("vitals:ffp", "ffp_transfusion"),
                ("vitals:pheresis", "platelet_transfusion"), ("vitals:cryo", "cryo_transfusion"),
            ):
                last = -math.inf
                for t, value in zip(*feature_arrays.get(feature, ([], []))):
                    if value > 0 and float(t) - last >= 30:
                        rec = outcome(event_name, float(t), float(value), 1)
                        if rec:
                            records.append(rec)
                        last = float(t)

            vasopressor_times = []
            for feature in ("vitals:epi", "vitals:nepi", "vitals:vaso", "vitals:eph", "vitals:phe", "vitals:dopai", "vitals:dobui"):
                if feature not in feature_arrays:
                    continue
                signal = feature_arrays[feature]
                signal_token = "med_signal:" + feature.split(":", 1)[1]
                if signal_token in token_vocab:
                    for dose_time, dose_value in zip(*signal):
                        if 0 <= float(dose_time) <= duration:
                            records.append(Record(
                                float(dose_time), signal_token,
                                TOKEN_KINDS["medication_context"],
                                value=float(dose_value), has_value=1))
                for transition in binary_transitions(*feature_arrays[feature], "vasopressor_start", "vasopressor_stop"):
                    rec = outcome(transition.name, transition.time_min, transition.value, 1)
                    if rec:
                        records.append(rec)
                    if transition.name == "vasopressor_start":
                        vasopressor_times.append(float(transition.time_min))

            last_context = {}
            last_vasopressor_medication = -math.inf
            for bin_value, kind, name in db.execute(
                "SELECT bin,kind,name FROM context WHERE ep=? ORDER BY bin", (ep,)
            ):
                t = float(bin_value) * 5.0
                if kind == "medication":
                    token, mapping_method = medication_resolver.resolve(name)
                    if token and t - last_context.get(token, -math.inf) >= 15:
                        records.append(Record(t, token, TOKEN_KINDS["medication_context"]))
                        last_context[token] = t
                        context_counts["medication_context"] += 1
                        medication_mapping[mapping_method] += 1
                    elif token:
                        medication_mapping["deduplicated_within_15_min"] += 1
                    elif not token:
                        dropped_context["unseen_medication"] += 1
                        medication_mapping["unmatched"] += 1
                        unmatched_medication_names[str(name)] += 1
                    if (any(term in str(name).lower() for term in VASOPRESSOR_TERMS) and
                            t - last_vasopressor_medication >= 30.0):
                        rec = outcome("vasopressor_start", t)
                        if rec:
                            records.append(rec)
                            vasopressor_times.append(t)
                            last_vasopressor_medication = t
                elif kind == "diagnosis":
                    key = normalise_text(name)
                    token = diagnosis_tokens.get(key)
                    if token:
                        records.append(Record(t, token, TOKEN_KINDS["diagnosis_context"]))
                        context_counts["diagnosis_context"] += 1
                    else:
                        dropped_context["unseen_diagnosis"] += 1

            for bin_value, name, value, has_value in db.execute(
                "SELECT bin,name,value,has_value FROM direct_event WHERE ep=? ORDER BY bin", (ep,)
            ):
                rec = outcome(name, float(bin_value) * 5.0, float(value), int(has_value))
                if rec:
                    records.append(rec)

            sbp = _series(feature_arrays, "vitals:art_sbp", "vitals:nibp_sbp", "ward_vitals:art_sbp", "ward_vitals:nibp_sbp")
            mbp = _series(feature_arrays, "vitals:art_mbp", "vitals:nibp_mbp", "ward_vitals:nibp_mbp")
            bp_times, bp_abnormality = _bp_abnormality(sbp, mbp)
            for vaso_time in sorted(set(vasopressor_times)):
                status, response_time = observed_vasopressor_response(vaso_time, bp_times, bp_abnormality)
                response_audit[status] += 1
                event_name = ("bp_recovered_after_vasopressor_observed" if status == "recovered" else
                              "hypotension_persistent_after_vasopressor_observed" if status == "persistent" else None)
                if event_name:
                    rec = outcome(event_name, response_time)
                    if rec:
                        records.append(rec)

            records.append(Record(duration, "<EPISODE_END>", TOKEN_KINDS["boundary"]))
            records = _dedupe([record for record in records if 0 <= record.time_min <= duration])
            if "<CLOCK>" in token_vocab:
                measurement_chunks = [times for times, _ in feature_arrays.values() if len(times)]
                measurement_times = (np.concatenate(measurement_chunks)
                                     if measurement_chunks else np.empty(0, np.float32))
                records = _insert_clock_tokens(
                    records, clock_interval, max_clock_per_gap, measurement_times)
            if add_phase_summaries:
                records = _insert_phase_summary_tokens(records)
            for record in records:
                if record.outcome >= 0:
                    outcome_counts[stored_outcomes[record.outcome]] += 1
            writers["token_id"].write(token_vocab[record.token] for record in records)
            writers["time_min"].write(record.time_min for record in records)
            writers["value"].write(record.value for record in records)
            writers["has_value"].write(record.has_value for record in records)
            writers["token_kind"].write(record.kind for record in records)
            writers["outcome_class"].write(record.outcome for record in records)
            total += len(records)
            ptr.append(total)
            if (position + 1) % 1000 == 0:
                print(json.dumps({"episodes": position + 1, "total": len(episodes), "tokens": total}), flush=True)
    finally:
        db.close()
        for writer in writers.values():
            writer.close()

    np.save(output_dir / "sequence_ptr.npy", np.asarray(ptr, dtype=np.int64))
    episodes.to_csv(output_dir / "admissions.csv", index=False)
    (output_dir / "token_vocabulary.json").write_text(json.dumps(token_vocab, ensure_ascii=False, indent=2))
    meta = {
        "version": train_meta.get("version", 4),
        "complete": True,
        "created_unix": time.time(),
        "source_name": source_name,
        "source_manifest": source_manifest,
        "split": "external_validation",
        "num_admissions": len(episodes),
        "training_unit": training_unit,
        "num_operation_episodes": (len(episodes) if training_unit == "operation_episode" else 0),
        "num_unique_admissions": int(episodes["admission_id"].nunique()),
        "num_tokens": total,
        "num_static": int(train_meta["num_static"]),
        "static_features": train_meta["static_features"],
        "trajectory_horizons_hours": train_meta["trajectory_horizons_hours"],
        "prediction_target": train_meta["prediction_target"],
        "token_vocabulary": token_vocab,
        "outcome_vocabulary": train_meta["outcome_vocabulary"],
        "outcome_family_vocabulary": train_meta.get("outcome_family_vocabulary", []),
        "outcome_to_family": train_meta.get("outcome_to_family", []),
        "stored_outcome_vocabulary": stored_outcomes,
        "outcome_class_remap": train_meta["outcome_class_remap"],
        "token_kinds": train_meta["token_kinds"],
        "outcome_definitions": train_meta["outcome_definitions"],
        "frozen_training_contract": str(Path(training_dir).resolve()),
        "normal_measurement_policy": train_meta["normal_measurement_policy"],
        "clock_token_policy": train_meta.get("clock_token_policy"),
        "phase_summary_tokens": train_meta.get("phase_summary_tokens", []),
        "physiology_summary_policy": train_meta.get("physiology_summary_policy", {}),
        "medication_response_semantics": train_meta["medication_response_semantics"],
        "audit_counts": dict(outcome_counts),
        "context_counts": dict(context_counts),
        "dropped_context_counts": dict(dropped_context),
        "medication_mapping_audit": {
            "method_counts": dict(medication_mapping),
            "matched_before_temporal_deduplication": int(
                medication_mapping["exact"] + medication_mapping["generic_prefix"] +
                medication_mapping["explicit_alias"] +
                medication_mapping["deduplicated_within_15_min"]),
            "unmatched": int(medication_mapping["unmatched"]),
            "top_unmatched_names": [
                {"name": name, "rows": int(amount)}
                for name, amount in unmatched_medication_names.most_common(50)
            ],
        },
        "vasopressor_response_audit": dict(response_audit),
        "source_audit": source_audit,
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (output_dir / "event_sequence_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta
