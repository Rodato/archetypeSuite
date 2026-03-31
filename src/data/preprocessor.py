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


class DataPreprocessor:

    def preprocess(
        self, df: pd.DataFrame, strategy: dict[str, Any]
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        df = df.copy()
        metadata: dict[str, Any] = {}

        # --- Drop columns ---
        drop_cols = strategy.get("drop_columns", [])
        existing_drop = [c for c in drop_cols if c in df.columns]
        df.drop(columns=existing_drop, inplace=True)
        metadata["columns_dropped"] = existing_drop

        # --- Impute missing values ---
        imputation = strategy.get("imputation", "mean")
        metadata["imputation_strategy"] = imputation

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if numeric_cols:
            num_imputer = SimpleImputer(strategy=imputation)
            df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

        if categorical_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

        # --- Encode categoricals ---
        encoding = strategy.get("encoding", "onehot")
        metadata["encoding_method"] = encoding
        encoded_columns: list[str] = []

        if categorical_cols:
            if encoding == "onehot":
                encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoded = encoder.fit_transform(df[categorical_cols])
                new_col_names = encoder.get_feature_names_out(categorical_cols).tolist()
                encoded_df = pd.DataFrame(encoded, columns=new_col_names, index=df.index)
                df = df.drop(columns=categorical_cols)
                df = pd.concat([df, encoded_df], axis=1)
                encoded_columns = new_col_names
            elif encoding == "label":
                for col in categorical_cols:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    encoded_columns.append(col)

        metadata["encoded_columns"] = encoded_columns

        # --- Scale numerics ---
        scaling = strategy.get("scaling", "standard")
        metadata["scaling_method"] = scaling

        scale_cols = df.select_dtypes(include="number").columns.tolist()
        if scale_cols:
            scaler_map = {
                "standard": StandardScaler,
                "minmax": MinMaxScaler,
                "robust": RobustScaler,
            }
            scaler_cls = scaler_map.get(scaling)
            if scaler_cls is None:
                raise ValueError(f"Unsupported scaling method: {scaling}")
            scaler = scaler_cls()
            df[scale_cols] = scaler.fit_transform(df[scale_cols])

        # --- Dimensionality reduction ---
        dim_red = strategy.get("dimensionality_reduction")
        if dim_red and dim_red.get("method") == "pca":
            n_components = dim_red["n_components"]
            pre_pca_columns = df.columns.tolist()
            pca = PCA(n_components=n_components)
            transformed = pca.fit_transform(df)
            pca_col_names = [f"PC{i+1}" for i in range(transformed.shape[1])]
            df = pd.DataFrame(transformed, columns=pca_col_names, index=df.index)
            metadata["pca_components"] = pca.components_.tolist()
            metadata["pca_explained_variance"] = pca.explained_variance_ratio_.tolist()
            metadata["pre_pca_columns"] = pre_pca_columns

        metadata["final_shape"] = df.shape

        return df, metadata
