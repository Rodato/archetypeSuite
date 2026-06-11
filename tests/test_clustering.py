import numpy as np
import pytest

from src.clustering.evaluator import ClusteringEvaluator
from src.clustering.executor import ClusteringExecutor
from src.clustering.registry import AlgorithmRegistry


class TestAlgorithmRegistry:
    def test_registry_contents(self):
        registry = AlgorithmRegistry()
        assert set(registry._algorithms) == {"KMeans", "AgglomerativeClustering"}

    def test_get_algorithm(self):
        registry = AlgorithmRegistry()
        entry = registry.get("KMeans")
        assert "class" in entry
        assert "default_params" in entry

    def test_get_missing(self):
        registry = AlgorithmRegistry()
        with pytest.raises(KeyError):
            registry.get("NonExistent")


class TestClusteringExecutor:
    def test_kmeans(self, numeric_df):
        registry = AlgorithmRegistry()
        executor = ClusteringExecutor(registry)
        result = executor.execute("KMeans", numeric_df.values, {"n_clusters": 3})
        assert len(result["labels"]) == len(numeric_df)
        assert result["algorithm"] == "KMeans"

    def test_agglomerative(self, numeric_df):
        registry = AlgorithmRegistry()
        executor = ClusteringExecutor(registry)
        result = executor.execute("AgglomerativeClustering", numeric_df.values, {"n_clusters": 3})
        assert len(result["labels"]) == len(numeric_df)
        assert result["algorithm"] == "AgglomerativeClustering"


class TestClusteringEvaluator:
    def test_evaluate(self, numeric_df):
        registry = AlgorithmRegistry()
        executor = ClusteringExecutor(registry)
        result = executor.execute("KMeans", numeric_df.values, {"n_clusters": 3})

        evaluator = ClusteringEvaluator()
        metrics = evaluator.evaluate(numeric_df.values, result["labels"])
        assert "silhouette_score" in metrics
        assert "calinski_harabasz_score" in metrics
        assert "davies_bouldin_score" in metrics
        assert metrics["n_clusters"] == 3

    def test_evaluate_single_cluster(self, numeric_df):
        evaluator = ClusteringEvaluator()
        labels = [0] * len(numeric_df)
        metrics = evaluator.evaluate(numeric_df.values, labels)
        assert metrics["silhouette_score"] is None
        assert "warning" in metrics

    def test_cluster_profiles(self, sample_df):
        labels = np.random.choice([0, 1, 2], len(sample_df))
        evaluator = ClusteringEvaluator()
        profiles = evaluator.compute_cluster_profiles(sample_df, labels)
        assert isinstance(profiles, dict)
        for cluster_id in profiles:
            assert "age" in profiles[cluster_id]
