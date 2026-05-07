"""Data loading and validation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .schema import TEMPERATURE_COLUMN, missing_columns, required_feature_columns


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Excel table and normalize column names."""
    table_path = Path(path)
    if not table_path.exists():
        raise FileNotFoundError(f"Input table does not exist: {table_path}")

    suffix = table_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(table_path)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(table_path)
    else:
        raise ValueError(
            f"Unsupported table format '{suffix}'. Use CSV, XLSX, or XLS."
        )

    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def filter_temperature(
    frame: pd.DataFrame, temperature_c: float | None, tolerance: float
) -> pd.DataFrame:
    """Filter rows by measurement temperature when requested."""
    if temperature_c is None:
        return frame
    if TEMPERATURE_COLUMN not in frame.columns:
        raise ValueError(
            f"--temperature requires a '{TEMPERATURE_COLUMN}' column in the data."
        )

    temperatures = pd.to_numeric(frame[TEMPERATURE_COLUMN], errors="coerce")
    mask = (temperatures - temperature_c).abs() <= tolerance
    filtered = frame.loc[mask].copy()
    if filtered.empty:
        raise ValueError(
            f"No rows found at {temperature_c:g} C within +/- {tolerance:g} C."
        )
    return filtered


def coerce_numeric_features(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Convert feature columns to numeric values, preserving missing cells for imputation."""
    features = frame.loc[:, feature_columns].copy()
    for column in feature_columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    return features


def build_training_arrays(
    frame: pd.DataFrame,
    target: str,
    include_temperature: bool,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series, list[str], pd.DataFrame]:
    """Validate a training table and return X plus log10-transformed y."""
    required_features = required_feature_columns(include_temperature)
    missing = missing_columns(set(frame.columns), required_features + [target])
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    working = frame.copy()
    working[target] = pd.to_numeric(working[target], errors="coerce")
    working = working.dropna(subset=[target])
    working = working.loc[working[target] > 0].copy()
    if working.empty:
        raise ValueError(
            f"Target column '{target}' has no positive numeric values. "
            "Resistivity must be positive because the model trains on log10(rho)."
        )

    features = coerce_numeric_features(working, required_features)
    y = working[target].astype(float)
    y_log10 = np.log10(y.to_numpy(dtype=float))
    return features, y_log10, y, required_features, working


def build_prediction_frame(
    frame: pd.DataFrame, feature_columns: list[str]
) -> pd.DataFrame:
    """Validate a candidate table and return numeric feature columns."""
    missing = missing_columns(set(frame.columns), feature_columns)
    if missing:
        raise ValueError("Missing required candidate columns: " + ", ".join(missing))
    return coerce_numeric_features(frame, feature_columns)
