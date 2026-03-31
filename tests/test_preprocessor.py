import pandas as pd

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

    def test_pca(self, numeric_df):
        preprocessor = DataPreprocessor()
        strategy = {
            "drop_columns": [],
            "imputation": "mean",
            "scaling": "standard",
            "encoding": "onehot",
            "dimensionality_reduction": {"method": "pca", "n_components": 2},
        }
        result_df, metadata = preprocessor.preprocess(numeric_df, strategy)
        assert result_df.shape[1] == 2
        assert "pca_explained_variance" in metadata

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
