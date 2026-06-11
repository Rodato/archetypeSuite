"""Tests de la evidencia diferenciadora por cluster (determinista, sin LLM)."""
from unittest.mock import patch

import pandas as pd

from src.core.evidence import cluster_evidence


def _df_with_planted_difference():
    # Cluster 0: horas altas y franja Madrugada; cluster 1: lo contrario.
    n = 40
    return pd.DataFrame({
        "horas": [7.0] * n + [1.5] * n,
        "edad": [25] * n + [40] * n,
        "franja": ["Madrugada"] * n + ["Mañana"] * n,
    }), [0] * n + [1] * n


class TestClusterEvidence:
    def test_cites_numeric_and_categorical_differentiators(self):
        df, labels = _df_with_planted_difference()
        out = cluster_evidence(df, labels)
        assert "### Cluster 0" in out and "### Cluster 1" in out
        assert "horas: 7.00 vs 4.25 global" in out  # media del grupo vs media total
        assert "σ)" in out
        assert "franja = Madrugada: 100% vs 50% global" in out

    def test_deterministic(self):
        df, labels = _df_with_planted_difference()
        assert cluster_evidence(df, labels) == cluster_evidence(df, labels)

    def test_fail_soft_on_mismatch(self):
        df, labels = _df_with_planted_difference()
        assert cluster_evidence(df, labels[:-1]) == "(sin evidencia adicional)"
        assert cluster_evidence(pd.DataFrame(), []) == "(sin evidencia adicional)"


class TestInterpretReceivesEvidence:
    def test_prompt_contains_evidence_block(self):
        from src.agents.nodes.interpret_node import _fallback_interpretation, interpret_node

        df, labels = _df_with_planted_difference()
        state = {
            "n_clusters": 2,
            "metrics": {"silhouette_score": 0.4},
            "cluster_profiles": {0: {}, 1: {}},
            "dataset_context": "test",
            "original_columns": list(df.columns),
            "raw_data": df.to_dict(orient="list"),
            "labels": labels,
        }
        captured = {}

        def fake_invoke(llm, prompt, schema, fallback):
            captured["prompt"] = prompt
            return _fallback_interpretation(2), None

        with patch("src.agents.nodes.interpret_node.get_narrative_llm", lambda: None), \
             patch("src.agents.nodes.interpret_node.invoke_json_with_retry", fake_invoke):
            interpret_node(state)

        assert "Evidencia diferenciadora por cluster" in captured["prompt"]
        assert "franja = Madrugada: 100% vs 50% global" in captured["prompt"]
        assert "ANCLA las narrativas en la evidencia" in captured["prompt"]
