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


class TestRefinementParamWhitelist:
    def test_drops_unknown_and_dangerous_params(self):
        from src.agents.nodes.refinement_node import _sanitize_suggested_params
        out = _sanitize_suggested_params({
            "random_state": None,   # rompería el determinismo
            "algorithm": "DBSCAN",  # kwarg válido de KMeans, valor inválido → crash en fit
            "n_clusters": 5,        # lo gestiona optimize_k, no el LLM
            "n_init": 20,
            "init": "random",
            "max_iter": 500,
        })
        assert out == {"n_init": 20, "init": "random", "max_iter": 500}

    def test_drops_wrong_types(self):
        from src.agents.nodes.refinement_node import _sanitize_suggested_params
        # bool es subclase de int; strings numéricos y valores de init inválidos tampoco pasan
        out = _sanitize_suggested_params({"n_init": True, "max_iter": "300", "init": "pca"})
        assert out == {}

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
