#!/usr/bin/env python3
"""Clean the surgery_part1 NTUH EEG/BIS subset and audit signal-only cohorts."""
from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


ROOT = Path(__file__).resolve().parents[1]
PART1 = ROOT / "data/surgery_part1/surgery"
OUT = ROOT / "data/surgery_part1/processed_signal_cohorts"
FS = 128.0


def clean_ntuh_eeg() -> dict:
    src = PART1 / "NTUH_EEG_BIS_24"
    dest = OUT / "ntuh_eeg_bis_24"
    eeg_out = dest / "eeg_float32_128hz_bandpass_0p5_40hz"
    eeg_out.mkdir(parents=True, exist_ok=True)
    rows, bis_rows = [], []
    sos = butter(4, (0.5, 40.0), btype="bandpass", fs=FS, output="sos")
    for path in sorted(src.glob("*.mat")):
        row = {"record_id": path.stem, "status": "excluded_qc", "reason": ""}
        try:
            with h5py.File(path, "r") as f:
                if "EEG" not in f or "bis" not in f:
                    raise ValueError("missing root EEG/bis datasets")
                eeg = np.asarray(f["EEG"][()]).reshape(-1).astype(np.float64)
                bis = np.asarray(f["bis"][()]).reshape(-1).astype(np.float64)
            if len(eeg) < int(FS * 60) or not len(bis):
                raise ValueError("recording shorter than 60s or empty BIS")
            good = np.isfinite(eeg)
            if not good.any() or (1.0 - good.mean()) > 0.001:
                raise ValueError("EEG nonfinite fraction exceeds 0.1%")
            repaired = int((~good).sum())
            if repaired:
                missing = np.flatnonzero(~good)
                eeg[missing] = np.interp(missing, np.flatnonzero(good), eeg[good])
            filtered = sosfiltfilt(sos, eeg).astype(np.float32)
            valid_bis = np.isfinite(bis) & (bis >= 0) & (bis <= 100)
            bis_clean = np.where(valid_bis, bis, np.nan).astype(np.float32)
            out_path = eeg_out / f"{path.stem}.npy"
            np.save(out_path, filtered, allow_pickle=False)
            reread = np.load(out_path, mmap_mode="r")
            if reread.dtype != np.float32 or len(reread) != len(eeg) or not np.isfinite(reread).all():
                raise ValueError("cleaned EEG readback validation failed")
            for i, value in enumerate(bis_clean):
                bis_rows.append({"record_id": path.stem, "bis_index": i,
                                 "relative_time_seconds": i * 5.0, "bis": value})
            row.update({"status": "included", "reason": "", "sampling_rate_hz": FS,
                        "eeg_samples": len(eeg), "duration_minutes": len(eeg) / FS / 60,
                        "eeg_nonfinite_repaired": repaired,
                        "eeg_raw_min": float(np.nanmin(eeg)), "eeg_raw_max": float(np.nanmax(eeg)),
                        "eeg_clean_min": float(filtered.min()), "eeg_clean_max": float(filtered.max()),
                        "bis_samples": len(bis), "bis_valid": int(valid_bis.sum()),
                        "bis_invalid": int((~valid_bis).sum()),
                        "cleaned_eeg": str(out_path.relative_to(OUT))})
        except Exception as exc:
            row["reason"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    dest.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest / "case_manifest.csv", index=False)
    pd.DataFrame(bis_rows).to_csv(dest / "bis_5s.csv", index=False)
    included = [r for r in rows if r["status"] == "included"]
    summary = {"source": "NTUH EEG+BIS 24 surgical records", "selected": len(rows),
               "included": len(included), "excluded_qc": len(rows) - len(included),
               "total_eeg_samples": sum(r.get("eeg_samples", 0) for r in included),
               "total_bis_samples": sum(r.get("bis_samples", 0) for r in included),
               "eeg_filter": "4th-order zero-phase Butterworth 0.5-40 Hz; native 128 Hz retained",
               "bis_rule": "retain finite values in [0,100]; invalid points set to NaN",
               "maomao_compatibility": "EEG/BIS are not clinical event fields in the frozen MAOMAO vocabulary; not scored"}
    (dest / "quality_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def audit_signal_only_sources() -> list[dict]:
    results = []
    bids = PART1 / "OpenNeuro_ds004541_EEG_fNIRS/on004541-v1.0.0.zip"
    with zipfile.ZipFile(bids) as z:
        files = [i for i in z.infolist() if not i.is_dir()]
        edf = [i for i in files if i.filename.lower().endswith(".edf")]
        snirf = [i for i in files if i.filename.lower().endswith(".snirf")]
        participants = [i for i in files if i.filename.endswith("participants.tsv")]
        results.append({"source": "OpenNeuro ds004541 EEG-fNIRS", "archive_members": len(files),
                        "eeg_edf": len(edf), "fnirs_snirf": len(snirf),
                        "nonempty_signal_files": sum(i.file_size > 0 for i in edf + snirf),
                        "participant_metadata_files": len(participants),
                        "cleaning_status": "container/BIDS inventory only",
                        "maomao_status": "excluded: EEG/fNIRS does not provide the frozen clinical-event inputs"})
    mgh = PART1 / "MGH_GABAergic_Anesthesia/ACCESS_REQUIRED.md"
    results.append({"source": "MGH GABAergic anesthesia", "local_data_files": 0,
                    "cleaning_status": "not available", "maomao_status": "excluded: access-controlled source not downloaded",
                    "access_note": str(mgh.relative_to(ROOT))})
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eeg = clean_ntuh_eeg()
    inventory = audit_signal_only_sources()
    report = {"ntuh_eeg_bis": eeg, "signal_only_cohorts": inventory}
    (OUT / "cohort_qc_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
