"""JSON-safe serialization helpers.

The pipeline produces numpy scalars, NaN/Inf floats and dict keys that are ints — none
of which survive a strict JSON encoder. `to_jsonable` walks any structure and returns a
value that `json.dumps(..., allow_nan=False)` accepts: numpy → native, NaN/Inf → None,
DataFrames/Series → plain Python, non-string dict keys → str.
"""
import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


def _clean_float(value: float):
    return float(value) if math.isfinite(value) else None


def to_jsonable(obj: Any) -> Any:
    """Recursively convert `obj` into something json.dumps can encode without NaN/numpy."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return _clean_float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return _clean_float(float(obj))
    if isinstance(obj, np.ndarray):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        # pd.NaT es instancia de datetime: sin este guard se serializa como "NaT".
        if pd.isna(obj):
            return None
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return [to_jsonable(x) for x in obj.tolist()]
    if isinstance(obj, pd.DataFrame):
        return dataframe_to_records(obj)
    if isinstance(obj, dict):
        return {_clean_key(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, np.generic):  # any remaining numpy scalar
        return to_jsonable(obj.item())
    # pandas NA / NaT
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def _clean_key(key: Any) -> str:
    if isinstance(key, str):
        return key
    if isinstance(key, (int, np.integer)):
        return str(int(key))
    return str(key)


def dataframe_to_records(df: pd.DataFrame) -> list:
    """DataFrame -> list of row dicts, fully JSON-safe."""
    records = df.to_dict(orient="records")
    return [to_jsonable(row) for row in records]


def dataframe_to_table(df: pd.DataFrame) -> dict:
    """DataFrame -> {columns, rows} preserving column order (robust for a table widget)."""
    columns = [str(c) for c in df.columns]
    rows = [[to_jsonable(v) for v in row] for row in df.itertuples(index=False, name=None)]
    return {"columns": columns, "rows": rows}
