"""Tests para guards y validaciones agregados en PR6 (robustez)."""
from unittest.mock import patch

import pandas as pd
import pytest

from src.agents.nodes.optimize_k_node import optimize_k_node


class TestOptimizeKGuards:
    def test_raises_on_too_few_rows(self):
        # 3 filas no es suficiente (mínimo: 4)
        state = {"processed_data": {"a": [1, 2, 3], "b": [4, 5, 6]}}
        with pytest.raises(ValueError, match="demasiado pequeño"):
            optimize_k_node(state)

    def test_raises_on_no_features(self):
        # DataFrame sin columnas (n_features=0)
        state = {"processed_data": {}}
        with pytest.raises(ValueError):
            optimize_k_node(state)

    def test_runs_with_minimum_viable_dataset(self):
        # 10 filas x 2 columnas — debería pasar
        state = {
            "processed_data": pd.DataFrame({
                "x": [0, 1, 0, 1, 0, 1, 5, 6, 5, 6],
                "y": [0, 0, 1, 1, 0, 1, 5, 5, 6, 6],
            }).to_dict(orient="list"),
        }
        result = optimize_k_node(state)
        assert "optimal_k" in result
        assert result["optimal_k"] >= 2


class TestRefinementGate:
    """Gate determinista: silhouette bajo el umbral → 1 reintento exhaustivo (sin LLM)."""

    def test_refines_once_below_threshold(self):
        from src.agents.nodes.refinement_node import refinement_node
        out = refinement_node({
            "refinement_count": 0,
            "metrics": {"silhouette_score": 0.12},
            "algorithm_params": {"n_clusters": 3},
        })
        assert out["should_refine"] is True
        assert out["algorithm_params"]["n_init"] == 30
        assert out["algorithm_params"]["n_clusters"] == 3  # se preserva lo previo
        assert "0.12" in out["refinement_reason"]

    def test_does_not_refine_above_threshold(self):
        from src.agents.nodes.refinement_node import refinement_node
        out = refinement_node({"refinement_count": 0, "metrics": {"silhouette_score": 0.4}})
        assert out["should_refine"] is False
        assert "aceptable" in out["refinement_reason"]

    def test_never_refines_twice(self):
        from src.agents.nodes.refinement_node import refinement_node
        out = refinement_node({"refinement_count": 1, "metrics": {"silhouette_score": 0.05}})
        assert out["should_refine"] is False

    def test_no_silhouette_no_refine(self):
        from src.agents.nodes.refinement_node import refinement_node
        out = refinement_node({"refinement_count": 0, "metrics": {}})
        assert out["should_refine"] is False

    def test_deterministic(self):
        from src.agents.nodes.refinement_node import refinement_node
        state = {"refinement_count": 0, "metrics": {"silhouette_score": 0.12}}
        assert refinement_node(dict(state)) == refinement_node(dict(state))


class TestExecutorSeed:
    def test_executor_reforces_random_state(self):
        from src.clustering.executor import ClusteringExecutor
        from src.clustering.registry import AlgorithmRegistry

        df = pd.DataFrame({
            "x": [0, 0, 1, 1, 5, 5, 6, 6],
            "y": [0, 1, 0, 1, 5, 6, 5, 6],
        })
        result = ClusteringExecutor(AlgorithmRegistry()).execute(
            "KMeans", df, {"n_clusters": 2, "random_state": 999}
        )
        assert result["params"]["random_state"] == 42


class TestNaTokens:
    def test_includes_pandas_defaults_and_spanish_sentinels(self):
        from src.data.ingest import NA_TOKENS
        # Defaults documentados de pandas (antes venían de una API privada)
        assert {"NULL", "N/A", "NaN", "<NA>", ""} <= NA_TOKENS
        # Centinelas ES/LatAm propios
        assert {"sin dato", "n/d", "no aplica"} <= NA_TOKENS


class TestProviderApiKeyGuard:
    def test_get_llm_json_raises_when_api_key_missing(self):
        from src.llm import provider
        with patch.object(provider.settings, "openrouter_api_key", ""):
            with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
                provider.get_llm_json()

    def test_get_narrative_llm_raises_when_api_key_missing(self):
        from src.llm import provider
        with patch.object(provider.settings, "openrouter_api_key", ""):
            with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
                provider.get_narrative_llm()

    def test_get_fast_text_llm_raises_when_api_key_missing(self):
        from src.llm import provider
        with patch.object(provider.settings, "openrouter_api_key", ""):
            with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
                provider.get_fast_text_llm()
