import json

from src.data.profiler import DataProfiler


class TestDataProfiler:
    def test_profile_shape(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df)
        assert profile["n_rows"] == 50
        assert profile["n_cols"] == 5

    def test_profile_columns(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df)
        assert len(profile["columns"]) == 5
        names = [c["name"] for c in profile["columns"]]
        assert "age" in names
        assert "region" in names

    def test_numeric_stats(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df)
        age_col = next(c for c in profile["columns"] if c["name"] == "age")
        assert age_col["is_numeric"]
        assert "mean" in age_col
        assert "std" in age_col
        assert "skewness" in age_col

    def test_categorical_stats(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df)
        region_col = next(c for c in profile["columns"] if c["name"] == "region")
        assert not region_col["is_numeric"]
        assert "top_categories" in region_col

    def test_json_serializable(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df)
        json.dumps(profile)  # should not raise

    def test_correlation_matrix(self, sample_df):
        profiler = DataProfiler()
        profile = profiler.profile(sample_df)
        assert "correlation_matrix" in profile
        assert isinstance(profile["correlation_matrix"], dict)
