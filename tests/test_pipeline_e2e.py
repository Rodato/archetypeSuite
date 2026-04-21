import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agents.graph import compile_graph
from src.config.settings import settings


def _mock_message(content: str) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    return msg


def _make_mock_llm(responses: list) -> MagicMock:
    llm = MagicMock()
    llm.invoke.side_effect = [_mock_message(r) for r in responses]
    return llm


def _preprocess_response() -> str:
    return json.dumps({
        "drop_columns": ["customer_id"],
        "imputation": "median",
        "scaling": "standard",
        "encoding": "onehot",
        "dimensionality_reduction": None,
        "reasoning": "Test fixture",
    })


def _refinement_response(should_refine: bool = False) -> str:
    return json.dumps({
        "should_refine": should_refine,
        "reason": "Métricas aceptables" if not should_refine else "Probar otra config",
        "suggested_algorithm": None,
        "suggested_params": None,
    })


def _interpret_response(n_clusters: int) -> str:
    archetypes = [
        {
            "cluster_id": i,
            "label": f"Arquetipo {i}",
            "description": f"Descripción del cluster {i}",
            "key_characteristics": [f"rasgo{j}" for j in range(3)],
            "differentiators": [f"dif{j}" for j in range(2)],
        }
        for i in range(n_clusters)
    ]
    return json.dumps({
        "archetypes": archetypes,
        "summary": "Resumen de prueba de la segmentación",
    })


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.read_csv("sample_data/customers.csv")


class TestPipelineEndToEnd:
    def test_full_pipeline_with_mocked_llms(self, sample_df):
        graph = compile_graph()

        json_llm = _make_mock_llm([
            _preprocess_response(),
            _refinement_response(should_refine=False),
        ])
        narrative_llm = _make_mock_llm([
            _interpret_response(n_clusters=10),
        ])

        initial_state = {
            "raw_data": sample_df.to_dict(orient="list"),
            "file_name": "customers.csv",
            "dataset_context": "Dataset de prueba",
            "refinement_count": 0,
            "log_messages": [],
        }

        with patch("src.agents.nodes.preprocess_node.get_llm_json", return_value=json_llm), \
             patch("src.agents.nodes.refinement_node.get_llm_json", return_value=json_llm), \
             patch("src.agents.nodes.interpret_node.get_narrative_llm", return_value=narrative_llm):
            final_state = None
            for state in graph.stream(initial_state, stream_mode="values"):
                final_state = state

        assert final_state is not None
        assert "labels" in final_state
        assert len(final_state["labels"]) == len(sample_df)
        assert final_state["n_clusters"] >= 2
        assert final_state["selected_algorithm"] == "KMeans"
        assert "archetypes" in final_state
        assert len(final_state["archetypes"]) >= 2
        assert "metrics" in final_state
        assert "silhouette_score" in final_state["metrics"]

    def test_refinement_respects_max_iterations(self, sample_df):
        graph = compile_graph()

        # LLM siempre pide refinar, para forzar el loop al máximo
        json_responses = [_preprocess_response()] + [_refinement_response(should_refine=True)] * 10
        narrative_responses = [_interpret_response(n_clusters=10)] * 10

        json_llm = _make_mock_llm(json_responses)
        narrative_llm = _make_mock_llm(narrative_responses)

        initial_state = {
            "raw_data": sample_df.to_dict(orient="list"),
            "file_name": "customers.csv",
            "refinement_count": 0,
            "log_messages": [],
        }

        with patch("src.agents.nodes.preprocess_node.get_llm_json", return_value=json_llm), \
             patch("src.agents.nodes.refinement_node.get_llm_json", return_value=json_llm), \
             patch("src.agents.nodes.interpret_node.get_narrative_llm", return_value=narrative_llm):
            final_state = None
            for state in graph.stream(initial_state, stream_mode="values"):
                final_state = state

        assert final_state["refinement_count"] <= settings.max_refinement_iterations + 1
