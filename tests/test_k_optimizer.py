import numpy as np
import pytest

from src.data.k_optimizer import KOptimizer


class TestKOptimizer:
    def test_analyze_returns_expected_keys(self, numeric_df):
        optimizer = KOptimizer(k_min=2, k_max=5)
        result = optimizer.analyze(numeric_df.values)

        assert "k_range" in result
        assert "inertias" in result
        assert "silhouette_scores" in result
        assert "best_silhouette_k" in result
        assert "best_silhouette_score" in result
        assert "elbow_k" in result
        assert "optimal_k" in result

    def test_k_range_is_correct(self, numeric_df):
        optimizer = KOptimizer(k_min=2, k_max=5)
        result = optimizer.analyze(numeric_df.values)
        assert result["k_range"] == [2, 3, 4, 5]

    def test_optimal_k_within_range(self, numeric_df):
        optimizer = KOptimizer(k_min=2, k_max=6)
        result = optimizer.analyze(numeric_df.values)
        assert result["optimal_k"] in result["k_range"]

    def test_elbow_k_within_range(self, numeric_df):
        optimizer = KOptimizer(k_min=2, k_max=6)
        result = optimizer.analyze(numeric_df.values)
        assert result["elbow_k"] in result["k_range"]

    def test_silhouette_scores_bounded(self, numeric_df):
        optimizer = KOptimizer(k_min=2, k_max=5)
        result = optimizer.analyze(numeric_df.values)
        for score in result["silhouette_scores"]:
            assert -1.0 <= score <= 1.0

    def test_inertias_decrease_monotonically(self, numeric_df):
        optimizer = KOptimizer(k_min=2, k_max=6)
        result = optimizer.analyze(numeric_df.values)
        inertias = result["inertias"]
        for i in range(len(inertias) - 1):
            assert inertias[i] >= inertias[i + 1]

    def test_k_max_capped_by_sample_size(self):
        small_data = np.random.randn(30, 2)
        optimizer = KOptimizer(k_min=2, k_max=10)
        result = optimizer.analyze(small_data)
        # k_max efectivo = min(10, 30//10) = 3
        assert max(result["k_range"]) <= 3

    def test_optimal_k_equals_best_silhouette_k(self, numeric_df):
        optimizer = KOptimizer(k_min=2, k_max=5)
        result = optimizer.analyze(numeric_df.values)
        assert result["optimal_k"] == result["best_silhouette_k"]
