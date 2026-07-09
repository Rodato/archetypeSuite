from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from src.config.settings import settings

# LLM aliases → valid SimpleImputer strategies ('mode' is the natural word but invalid).
_IMPUTATION_ALIAS = {
    "mode": "most_frequent",
    "most_frequent": "most_frequent",
    "mean": "mean",
    "median": "median",
}


class DataPreprocessor:

    def preprocess(
        self, df: pd.DataFrame, strategy: dict[str, Any]
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        df = df.copy()
        metadata: dict[str, Any] = {}

        # --- Drop requested columns + any all-NaN columns (can't impute, would poison KMeans) ---
        drop_cols = strategy.get("drop_columns", []) or []
        existing_drop = [c for c in drop_cols if c in df.columns]
        df.drop(columns=existing_drop, inplace=True)
        all_nan = [c for c in df.columns if df[c].isna().all()]
        if all_nan:
            df.drop(columns=all_nan, inplace=True)
        metadata["columns_dropped"] = existing_drop + all_nan

        # Booleans behave as 2-level categoricals.
        for col in df.select_dtypes(include="bool").columns:
            df[col] = df[col].astype(str)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        # Incluye "string" (StringDtype de pandas): sin esto, una columna dtype='string' se
        # saltea de categorical_cols → ni one-hot ni ordinal → strings crudos a KMeans → crash.
        categorical_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

        # --- Ordinal-en-texto: mapear por el orden curado → rango entero (feature numérica) ---
        # Vacío por defecto ⇒ toda categórica es nominal (one-hot). El mapa lo provee el wizard
        # (paso 2); acá solo se aplica. La columna pasa a numérica: se imputa (median) y se escala,
        # preservando el gradiente ordinal en la distancia de KMeans (a diferencia de one-hot).
        ordinal_mappings = strategy.get("ordinal_mappings") or {}
        ordinal_applied: list[str] = []
        for col in sorted(ordinal_mappings, key=str):  # orden estable → metadata/log deterministas
            order = ordinal_mappings[col]
            if col not in categorical_cols or not isinstance(order, (list, tuple)) or not order:
                continue
            # Dedup por PRIMERA aparición: un orden con duplicados no debe invertir la escala
            # (con dict-comprehension la última aparición pisaba y daba Bajo>Medio).
            rank: dict[str, int] = {}
            for cat in order:
                rank.setdefault(str(cat), len(rank))
            mapped = df[col].astype(str).map(rank)  # desconocidos/faltantes → NaN (los cubre la imputación)
            if mapped.notna().any():  # al menos un valor cae dentro del orden dado
                df[col] = mapped.astype(float)
                categorical_cols.remove(col)
                numeric_cols.append(col)
                ordinal_applied.append(col)
        metadata["ordinal_encoded"] = ordinal_applied

        # --- Impute missing values ---
        raw_imp = str(strategy.get("imputation", "median")).lower()
        imputation = _IMPUTATION_ALIAS.get(raw_imp, "median")
        metadata["imputation_strategy"] = imputation

        if numeric_cols:
            num_imputer = SimpleImputer(strategy=imputation)
            df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

        if categorical_cols:
            # Preserve a genuine "nan" category: only treat actual NaN as missing.
            for col in categorical_cols:
                mask = df[col].isna()
                df[col] = df[col].astype(str)
                if mask.any():
                    df.loc[mask, col] = pd.NA
            cat_imputer = SimpleImputer(strategy="most_frequent")
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

        # --- Encode categoricals (one-hot capped so high-cardinality doesn't explode dims) ---
        encoding = strategy.get("encoding", "onehot")
        metadata["encoding_method"] = encoding
        encoded_columns: list[str] = []

        if categorical_cols:
            if encoding == "label":
                for col in categorical_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    encoded_columns.append(col)
            else:  # one-hot, capped
                encoder = OneHotEncoder(
                    sparse_output=False,
                    handle_unknown="infrequent_if_exist",
                    max_categories=settings.max_onehot_levels,
                )
                encoded = encoder.fit_transform(df[categorical_cols])
                new_col_names = encoder.get_feature_names_out(categorical_cols).tolist()
                encoded_df = pd.DataFrame(encoded, columns=new_col_names, index=df.index)
                df = df.drop(columns=categorical_cols)
                df = pd.concat([df, encoded_df], axis=1)
                encoded_columns = new_col_names

        metadata["encoded_columns"] = encoded_columns

        # --- Scale ONLY the genuine continuous columns (never the 0/1 one-hot dummies) ---
        scaling = strategy.get("scaling", "standard")
        metadata["scaling_method"] = scaling
        scale_cols = [c for c in numeric_cols if c in df.columns]
        if scale_cols:
            scaler_map = {
                "standard": StandardScaler,
                "minmax": MinMaxScaler,
                "robust": RobustScaler,
            }
            scaler_cls = scaler_map.get(scaling, StandardScaler)
            scaler = scaler_cls()
            df[scale_cols] = scaler.fit_transform(df[scale_cols])

        # --- Dimensionality reduction (PCA) — OFF by default (see settings.enable_pca) ---
        dim_red = strategy.get("dimensionality_reduction")
        if dim_red and dim_red.get("method") == "pca":
            if not settings.enable_pca:
                metadata["pca_skipped"] = "deshabilitado (evita el colapso a 1-D)"
            else:
                n_components = _safe_n_components(dim_red.get("n_components"), df.shape)
                if n_components:
                    pre_pca_columns = df.columns.tolist()
                    pca = PCA(n_components=n_components)
                    transformed = pca.fit_transform(df)
                    pca_col_names = [f"PC{i+1}" for i in range(transformed.shape[1])]
                    df = pd.DataFrame(transformed, columns=pca_col_names, index=df.index)
                    metadata["pca_explained_variance"] = pca.explained_variance_ratio_.tolist()
                    metadata["pre_pca_columns"] = pre_pca_columns

        metadata["final_shape"] = df.shape
        return df, metadata


def _safe_n_components(n_components, shape):
    """Clamp PCA components: float variance-ratio kept as-is; ints floored at 2."""
    max_comp = max(1, min(shape[1] - 1, shape[0] - 1))
    if isinstance(n_components, float) and 0.0 < n_components < 1.0:
        return n_components
    try:
        n = int(n_components)
    except (TypeError, ValueError):
        n = max_comp
    return max(2, min(n, max_comp)) if max_comp >= 2 else None
