import numpy as np
import pandas as pd


def _to_native(val):
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val) if not np.isnan(val) else None
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if pd.isna(val):
        return None
    return val


class DataProfiler:

    def profile(self, df: pd.DataFrame) -> dict:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        columns_info: list[dict] = []
        for col in df.columns:
            info: dict = {
                "name": col,
                "dtype": str(df[col].dtype),
                "n_missing": int(df[col].isna().sum()),
                "pct_missing": float(df[col].isna().mean()),
                "n_unique": int(df[col].nunique()),
                "is_numeric": col in numeric_cols,
            }

            if col in numeric_cols:
                info["mean"] = _to_native(df[col].mean())
                info["std"] = _to_native(df[col].std())
                info["min"] = _to_native(df[col].min())
                info["max"] = _to_native(df[col].max())
                info["median"] = _to_native(df[col].median())
                info["skewness"] = _to_native(df[col].skew())
                info["q1"] = _to_native(df[col].quantile(0.25))
                info["q3"] = _to_native(df[col].quantile(0.75))

            if col in categorical_cols:
                top = df[col].value_counts().head(10)
                info["top_categories"] = {
                    str(k): int(v) for k, v in top.items()
                }

            columns_info.append(info)

        corr = df[numeric_cols].corr() if numeric_cols else pd.DataFrame()
        corr_dict: dict[str, dict[str, float]] = {
            str(outer_k): {str(inner_k): _to_native(inner_v) for inner_k, inner_v in outer_v.items()}
            for outer_k, outer_v in corr.to_dict().items()
        }

        return {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": columns_info,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "correlation_matrix": corr_dict,
        }
