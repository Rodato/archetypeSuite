import pandas as pd

from src.config.settings import settings
from src.data.preprocessor import DataPreprocessor


class TestDataPreprocessor:
    def test_basic_preprocessing(self, sample_df):
        preprocessor = DataPreprocessor()
        strategy = {
            "drop_columns": [],
            "imputation": "mean",
            "scaling": "standard",
            "encoding": "onehot",
            "dimensionality_reduction": None,
        }
        result_df, metadata = preprocessor.preprocess(sample_df, strategy)
        assert not result_df.empty
        assert "scaling_method" in metadata
        assert metadata["scaling_method"] == "standard"

    def test_drop_columns(self, sample_df):
        preprocessor = DataPreprocessor()
        strategy = {
            "drop_columns": ["region"],
            "imputation": "median",
            "scaling": "minmax",
            "encoding": "onehot",
            "dimensionality_reduction": None,
        }
        result_df, metadata = preprocessor.preprocess(sample_df, strategy)
        assert "region" not in result_df.columns
        assert "region" in metadata["columns_dropped"]

    def test_label_encoding(self, sample_df):
        preprocessor = DataPreprocessor()
        strategy = {
            "drop_columns": [],
            "imputation": "mean",
            "scaling": "standard",
            "encoding": "label",
            "dimensionality_reduction": None,
        }
        result_df, metadata = preprocessor.preprocess(sample_df, strategy)
        assert metadata["encoding_method"] == "label"
        # After label encoding + scaling, dtype is float
        assert result_df["region"].dtype in ("int64", "int32", "float64")

    def test_pca_disabled_by_default(self, numeric_df):
        # PCA is off by default — data passes through unreduced (avoids the 1-D collapse
        # that inflates silhouette). The requested reduction is ignored and noted.
        preprocessor = DataPreprocessor()
        strategy = {
            "drop_columns": [],
            "imputation": "mean",
            "scaling": "standard",
            "encoding": "onehot",
            "dimensionality_reduction": {"method": "pca", "n_components": 2},
        }
        result_df, metadata = preprocessor.preprocess(numeric_df, strategy)
        assert result_df.shape[1] == numeric_df.shape[1]
        assert "pca_skipped" in metadata

    def test_pca_when_enabled_clamps_to_min_two(self, numeric_df, monkeypatch):
        monkeypatch.setattr(settings, "enable_pca", True)
        preprocessor = DataPreprocessor()
        strategy = {
            "drop_columns": [],
            "imputation": "mean",
            "scaling": "standard",
            "encoding": "onehot",
            "dimensionality_reduction": {"method": "pca", "n_components": 1},
        }
        result_df, metadata = preprocessor.preprocess(numeric_df, strategy)
        # n_components=1 is clamped up to 2 (never collapse to a single axis)
        assert result_df.shape[1] == 2
        assert "pca_explained_variance" in metadata

    def test_mode_imputation_is_aliased(self, sample_df):
        # 'mode' is the natural word but invalid for SimpleImputer → must map to most_frequent.
        df = sample_df.copy()
        df.loc[0, "age"] = None
        preprocessor = DataPreprocessor()
        strategy = {
            "drop_columns": [],
            "imputation": "mode",
            "scaling": "standard",
            "encoding": "onehot",
            "dimensionality_reduction": None,
        }
        result_df, metadata = preprocessor.preprocess(df, strategy)
        assert metadata["imputation_strategy"] == "most_frequent"
        assert result_df.isnull().sum().sum() == 0

    def test_with_missing_values(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "age"] = None
        df.loc[1, "income"] = None
        preprocessor = DataPreprocessor()
        strategy = {
            "drop_columns": [],
            "imputation": "median",
            "scaling": "robust",
            "encoding": "onehot",
            "dimensionality_reduction": None,
        }
        result_df, metadata = preprocessor.preprocess(df, strategy)
        assert result_df.isnull().sum().sum() == 0
