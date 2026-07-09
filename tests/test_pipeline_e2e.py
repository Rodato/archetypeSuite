import json
from unittest.mock import MagicMock, patch

import numpy as np
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


def _upstream_selection(df: pd.DataFrame) -> dict:
    """Estado que dispara el fast path de column_selection (sin LLM) → e2e hermético."""
    cols = list(df.columns)
    return {
        "selected_columns": cols,
        "static_filter_result": {"kept": cols, "dropped": [], "datetime_extracted": []},
        "column_recommendation": {"selected_columns": [], "excluded_columns": [], "summary": "test"},
    }


def _interpret_response(n_clusters: int) -> str:
    archetypes = [
        {
            "cluster_id": i,
            "label": f"Patrón {i}",
            "description": f"En este grupo aparece un patrón de prueba {i}.",
            "comportamiento_principal": f"Conducta distintiva {i}.",
            "microcomportamientos": [f"micro{j}" for j in range(3)],
            "barreras": [f"barrera{j} (motivación automática)" for j in range(2)],
            "habilitadores": [f"habilitador{j}" for j in range(2)],
            "oportunidades_accion": [f"Explorar {j}" for j in range(2)],
            "nivel_cautela": "baja",  # claimed low — the deterministic floor may raise it
            "cautela_reason": "Lectura de prueba.",
        }
        for i in range(n_clusters)
    ]
    return json.dumps({
        "archetypes": archetypes,
        "summary": "Resumen de prueba de la segmentación en clave de patrones.",
    })


@pytest.fixture
def sample_df() -> pd.DataFrame:
    # Sintético determinista (antes leía customers.csv, eliminado con el legacy Streamlit).
    rng = np.random.default_rng(42)
    n = 50
    return pd.DataFrame({
        "edad": rng.integers(18, 70, n),
        "ingreso": rng.integers(20_000, 120_000, n),
        "gasto_mensual": rng.integers(100, 5_000, n),
        "visitas_mes": rng.integers(1, 30, n),
        "ciudad": rng.choice(["Bogotá", "Lima", "Quito"], n),
        "segmento": rng.choice(["Básico", "Premium"], n),
    })


class TestPipelineEndToEnd:
    def test_full_pipeline_with_mocked_llms(self, sample_df):
        graph = compile_graph()

        # preprocess ya es determinista (sin LLM) → solo se mockea la capa narrativa (interpret).
        narrative_llm = _make_mock_llm([
            _interpret_response(n_clusters=10),
        ])

        initial_state = {
            "raw_data": sample_df.to_dict(orient="list"),
            "file_name": "customers.csv",
            "dataset_context": "Dataset de prueba",
            "refinement_count": 0,
            "log_messages": [],
            # Fast path de column_selection (sin LLM) → e2e hermético, sin red.
            **_upstream_selection(sample_df),
        }

        with patch("src.agents.nodes.interpret_node.get_narrative_llm", return_value=narrative_llm):
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

        # Behavioral layer flows through + summary is captured.
        assert "interpretation_summary" in final_state
        archs = final_state["archetypes"]
        assert all("nivel_cautela" in a for a in archs)
        assert all(a["nivel_cautela"] in ("baja", "media", "alta") for a in archs)
        assert all("barreras" in a for a in archs)

        # Deterministic caution floor (§9): no archetype may sit below the silhouette-derived floor.
        from src.core.quality import CAUTION_ORDER, caution_from_silhouette
        floor = caution_from_silhouette(final_state["metrics"]["silhouette_score"])
        assert all(CAUTION_ORDER[a["nivel_cautela"]] >= CAUTION_ORDER[floor] for a in archs)

    def test_refinement_respects_max_iterations(self, sample_df):
        graph = compile_graph()

        # El gate determinista decide el refinamiento según silhouette (sin LLM):
        # con datos aleatorios de 50 filas la separación es débil → refina 1 vez máximo.
        narrative_llm = _make_mock_llm([_interpret_response(n_clusters=10)] * 10)

        initial_state = {
            "raw_data": sample_df.to_dict(orient="list"),
            "file_name": "customers.csv",
            "refinement_count": 0,
            "log_messages": [],
            **_upstream_selection(sample_df),
        }

        with patch("src.agents.nodes.interpret_node.get_narrative_llm", return_value=narrative_llm):
            final_state = None
            for state in graph.stream(initial_state, stream_mode="values"):
                final_state = state

        assert final_state["refinement_count"] <= settings.max_refinement_iterations + 1
