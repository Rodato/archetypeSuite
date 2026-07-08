from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.agents.nodes.column_selection_node import column_selection_node, suggest_columns
from src.models.schemas import ColumnRecommendation, ColumnRelevanceDecision


@pytest.fixture
def messy_df():
    np.random.seed(0)
    n = 30
    return pd.DataFrame({
        "customer_id": list(range(n)),
        "country": ["MX"] * n,
        "comment": [
            "Comentario muy largo del cliente sobre su experiencia con el producto"
        ] * n,
        "age": np.random.randint(20, 60, n),
        "income": np.random.randint(20000, 100000, n),
        "spending_score": np.random.randint(1, 100, n),
        "region": np.random.choice(["North", "South"], n),
    })


def _mock_invoke_json_with_retry(decision: ColumnRelevanceDecision):
    return lambda *_args, **_kwargs: (decision, None)


def test_suggest_columns_uses_llm_picks(messy_df):
    fake_decision = ColumnRelevanceDecision(
        selected_columns=[
            ColumnRecommendation(name="age", reason="demografía", importance="medium"),
            ColumnRecommendation(name="income", reason="poder adquisitivo", importance="high"),
            ColumnRecommendation(name="spending_score", reason="comportamiento", importance="high"),
        ],
        excluded_columns=[],
        summary="Selección basada en variables de comportamiento.",
    )

    with patch("src.agents.nodes.column_selection_node.invoke_json_with_retry", _mock_invoke_json_with_retry(fake_decision)), \
         patch("src.agents.nodes.column_selection_node.get_llm_json", lambda: None):
        result = suggest_columns(messy_df, dataset_context="Clientes retail.")

    assert "customer_id" not in result["filtered_df"].columns  # caught by static filters
    assert "country" not in result["filtered_df"].columns
    assert "comment" not in result["filtered_df"].columns
    assert {r["name"] for r in result["column_recommendation"]["selected_columns"]} == {
        "age", "income", "spending_score"
    }
    assert result["llm_error"] is None


def test_suggest_columns_falls_back_when_llm_invents_columns(messy_df):
    fake_decision = ColumnRelevanceDecision(
        selected_columns=[
            ColumnRecommendation(name="ESTA_NO_EXISTE", reason="x", importance="medium"),
        ],
        excluded_columns=[],
        summary="—",
    )

    with patch("src.agents.nodes.column_selection_node.invoke_json_with_retry", _mock_invoke_json_with_retry(fake_decision)), \
         patch("src.agents.nodes.column_selection_node.get_llm_json", lambda: None):
        result = suggest_columns(messy_df, dataset_context=None)

    selected = {r["name"] for r in result["column_recommendation"]["selected_columns"]}
    available = set(result["filtered_df"].columns)
    assert selected == available  # fallback returns all post-filter columns


def test_node_respects_upstream_selection(messy_df):
    state = {
        "raw_data": messy_df.to_dict(orient="list"),
        "selected_columns": ["age", "income"],
        "static_filter_result": {
            "kept": ["age", "income", "spending_score", "region"],
            "dropped": [{"column": "customer_id", "reason": "Identificador"}],
            "datetime_extracted": [],
        },
        "column_recommendation": {
            "selected_columns": [{"name": "age", "reason": "x", "importance": "medium"}],
            "excluded_columns": [],
            "summary": "—",
        },
    }
    out = column_selection_node(state)
    assert out["selected_columns"] == ["age", "income"]
    assert set(out["raw_data"].keys()) == {"age", "income"}


def test_node_runs_llm_when_no_upstream(messy_df):
    fake_decision = ColumnRelevanceDecision(
        selected_columns=[
            ColumnRecommendation(name="age", reason="x", importance="high"),
            ColumnRecommendation(name="region", reason="y", importance="medium"),
        ],
        excluded_columns=[],
        summary="—",
    )
    state = {
        "raw_data": messy_df.to_dict(orient="list"),
        "dataset_context": "Test",
    }
    with patch("src.agents.nodes.column_selection_node.invoke_json_with_retry", _mock_invoke_json_with_retry(fake_decision)), \
         patch("src.agents.nodes.column_selection_node.get_llm_json", lambda: None):
        out = column_selection_node(state)

    assert out["selected_columns"] == ["age", "region"]
    assert set(out["raw_data"].keys()) == {"age", "region"}
    assert any("LLM sugiere" in msg for msg in out["log_messages"])


def test_node_raises_clear_error_when_all_columns_filtered():
    # id + constante + texto libre → los filtros estáticos descartan todo → 0 columnas.
    # Antes: raw_data={} perdía las filas y optimize_k mentía "0 filas"; ahora: mensaje preciso.
    df = pd.DataFrame({
        "user_id": list(range(6)),
        "country": ["MX"] * 6,
        "notes": ["comentario larguísimo del usuario sobre el producto y su experiencia"] * 6,
    })
    with pytest.raises(ValueError, match="ninguna columna utilizable"):
        column_selection_node({"raw_data": df.to_dict(orient="list"), "dataset_context": "x"})


def test_node_raises_clear_error_fast_path_when_all_columns_filtered():
    # Fast path (upstream selection presente) donde los filtros dejan 0 columnas → mismo guard.
    df = pd.DataFrame({
        "user_id": list(range(6)),
        "country": ["MX"] * 6,
        "notes": ["comentario larguísimo del usuario sobre el producto y su experiencia"] * 6,
    })
    state = {
        "raw_data": df.to_dict(orient="list"),
        "selected_columns": ["user_id"],
        "static_filter_result": {"kept": [], "dropped": [], "datetime_extracted": []},
        "column_recommendation": {"selected_columns": [], "excluded_columns": [], "summary": "—"},
    }
    with pytest.raises(ValueError, match="ninguna columna utilizable"):
        column_selection_node(state)
