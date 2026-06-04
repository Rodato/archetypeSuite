"""Static column filtering for clustering pipeline.

Detects and excludes columns that are unsuitable for clustering by deterministic
heuristics (IDs, near-zero variance, free-text, high missing, datetimes).
Datetime columns are extracted into numeric features (year) rather than dropped.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ID_NAME_PATTERN = re.compile(r"^(id|uuid|.*_id|.*_uuid|.*id\d*)$", re.IGNORECASE)

FREE_TEXT_MEAN_LEN = 50
FREE_TEXT_UNIQUE_RATIO = 0.5
HIGH_MISSING_THRESHOLD = 0.7
NEAR_ZERO_NUMERIC_CV = 0.01


def detect_id_column(df: pd.DataFrame, col: str) -> bool:
    if ID_NAME_PATTERN.match(col):
        return True
    n_rows = len(df)
    if n_rows == 0:
        return False
    series = df[col]
    if series.nunique(dropna=True) != n_rows:
        return False
    if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        return True
    if pd.api.types.is_integer_dtype(series):
        sorted_vals = series.dropna().sort_values().to_numpy()
        if len(sorted_vals) > 1:
            diffs = np.diff(sorted_vals)
            if np.all(diffs == 1):
                return True
    return False


def detect_near_zero_variance(df: pd.DataFrame, col: str) -> bool:
    series = df[col]
    if series.nunique(dropna=True) <= 1:
        return True
    if pd.api.types.is_numeric_dtype(series):
        std = float(series.std()) if len(series) > 1 else 0.0
        mean = float(series.mean())
        if mean != 0 and abs(std / mean) < NEAR_ZERO_NUMERIC_CV:
            return True
    return False


def detect_free_text(df: pd.DataFrame, col: str) -> bool:
    series = df[col]
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
        return False
    n_rows = len(series.dropna())
    if n_rows == 0:
        return False
    str_series = series.dropna().astype(str)
    mean_len = str_series.str.len().mean()
    unique_ratio = str_series.nunique() / n_rows
    if mean_len > FREE_TEXT_MEAN_LEN:
        return True
    if unique_ratio > FREE_TEXT_UNIQUE_RATIO and mean_len > 15:
        return True
    return False


def detect_high_missing(df: pd.DataFrame, col: str, threshold: float = HIGH_MISSING_THRESHOLD) -> bool:
    return float(df[col].isna().mean()) > threshold


def _try_parse_datetime(series: pd.Series) -> Optional[pd.Series]:
    if not pd.api.types.is_object_dtype(series) and not pd.api.types.is_string_dtype(series):
        return None
    sample = series.dropna().head(20)
    if len(sample) < 3:
        return None
    try:
        parsed = pd.to_datetime(sample, errors="raise")
    except (ValueError, TypeError):
        return None
    if parsed.dt.year.nunique() == 1 and parsed.dt.month.nunique() == 1 and parsed.dt.day.nunique() == 1:
        return None
    try:
        return pd.to_datetime(series, errors="coerce")
    except (ValueError, TypeError):
        return None


def detect_datetime(df: pd.DataFrame, col: str) -> bool:
    if pd.api.types.is_datetime64_any_dtype(df[col]):
        return True
    return _try_parse_datetime(df[col]) is not None


def extract_datetime_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Replace a datetime column with a numeric `{col}_year` feature.

    Returns a copy of df with the datetime column dropped and the year added.
    """
    out = df.copy()
    if pd.api.types.is_datetime64_any_dtype(out[col]):
        parsed = out[col]
    else:
        parsed = _try_parse_datetime(out[col])
        if parsed is None:
            return out
    out[f"{col}_year"] = parsed.dt.year.astype("Int64").astype("float64")
    out.drop(columns=[col], inplace=True)
    return out


def apply_static_filters(
    df: pd.DataFrame,
    *,
    extract_datetime: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply heuristic filters to drop unusable columns and extract datetime features.

    Returns (filtered_df, report) where report has keys:
      - kept: list of remaining column names
      - dropped: list of {column, reason}
      - datetime_extracted: list of {original, new}
    """
    out = df.copy()
    dropped: List[Dict[str, str]] = []
    datetime_extracted: List[Dict[str, str]] = []

    cols = list(out.columns)
    for col in cols:
        if col not in out.columns:
            continue

        if detect_high_missing(out, col):
            dropped.append({"column": col, "reason": "Más del 70% de valores faltantes"})
            out.drop(columns=[col], inplace=True)
            continue

        if detect_near_zero_variance(out, col):
            dropped.append({"column": col, "reason": "Sin variación (todos los valores son iguales o casi iguales)"})
            out.drop(columns=[col], inplace=True)
            continue

        if detect_datetime(out, col):
            if extract_datetime:
                out = extract_datetime_features(out, col)
                datetime_extracted.append({"original": col, "new": f"{col}_year"})
            else:
                dropped.append({"column": col, "reason": "Columna de fecha (no se extrajeron features)"})
                out.drop(columns=[col], inplace=True)
            continue

        if detect_id_column(out, col):
            dropped.append({"column": col, "reason": "Identificador único (no aporta a la segmentación)"})
            out.drop(columns=[col], inplace=True)
            continue

        if detect_free_text(out, col):
            dropped.append({"column": col, "reason": "Texto libre de alta cardinalidad"})
            out.drop(columns=[col], inplace=True)
            continue

    report = {
        "kept": list(out.columns),
        "dropped": dropped,
        "datetime_extracted": datetime_extracted,
    }
    return out, report
