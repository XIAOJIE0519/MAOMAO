"""Shared admission-level static feature construction for train/external data."""
from __future__ import annotations

import numpy as np
import pandas as pd


STATIC_FEATURES = (
    "age_at_operation", "male", "bmi", "asa", "emergency",
    "weight_kg", "height_cm",
)


def _column(frame: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _emergency(frame: pd.DataFrame) -> pd.Series:
    if "emop" not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    raw = frame["emop"]
    numeric = pd.to_numeric(raw, errors="coerce")
    text = raw.astype("string").str.strip().str.lower()
    result = numeric.copy()
    result = result.where(result.isin([0, 1]), np.nan)
    emergency_words = text.isin({"1", "true", "yes", "urgent", "emergency", "emergent", "急诊"})
    elective_words = text.isin({"0", "false", "no", "elective", "择期"})
    result = result.mask(emergency_words, 1.0).mask(elective_words, 0.0)
    return result


def build_static_features(frame: pd.DataFrame, *, age_names=("age",),
                          male_names=("male",)) -> np.ndarray:
    """Return the fixed seven-feature matrix used by every dataset.

    Missing values are represented as zero after scaling; the corresponding
    information is not fabricated.  Existing dynamic context tokens remain in
    the sequence, so this matrix only contains values known at admission or
    operation start.
    """
    age = _column(frame, *age_names).fillna(0.0) / 100.0
    male = _column(frame, *male_names).fillna(0.0)
    weight = _column(frame, "weight", "weight_kg").clip(lower=0, upper=300)
    height = _column(frame, "height", "height_cm").clip(lower=0, upper=250)
    bmi = weight / (height / 100.0).pow(2)
    bmi = bmi.where(np.isfinite(bmi), np.nan).clip(lower=0, upper=80).fillna(0.0) / 40.0
    asa = _column(frame, "asa").clip(lower=0, upper=6).fillna(0.0) / 6.0
    emergency = _emergency(frame).fillna(0.0)
    matrix = np.column_stack((
        age.to_numpy(np.float32),
        male.clip(lower=0, upper=1).to_numpy(np.float32),
        bmi.to_numpy(np.float32),
        asa.to_numpy(np.float32),
        emergency.clip(lower=0, upper=1).to_numpy(np.float32),
        (weight.fillna(0.0) / 150.0).to_numpy(np.float32),
        (height.fillna(0.0) / 200.0).to_numpy(np.float32),
    )).astype(np.float32)
    return matrix
