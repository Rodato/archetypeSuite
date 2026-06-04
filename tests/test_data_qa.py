from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.llm.data_qa import _execute, _format_history, _validate_columns, answer_data_question
from src.models.schemas import DataQuery


@pytest.fixture
def chat_df():
    np.random.seed(0)
    n = 30
    return pd.DataFrame({
        "age": np.random.randint(20, 70, n),
        "income": np.random.randint(20000, 100000, n),
        "gender": np.random.choice(["F", "M"], n),
        "region": np.random.choice(["North", "South", "East"], n),
    })


def _patch_llm(query: DataQuery, error=None):
    return (
        patch(
            "src.llm.data_qa.invoke_json_with_retry",
            lambda *_args, **_kwargs: (query, error),
        ),
        patch("src.llm.data_qa.get_llm_json", lambda: None),
        patch("src.llm.data_qa._generate_natural_narrative", lambda *_a, **_kw: None),
    )


def test_filter_count_returns_total_rows(chat_df):
    query = DataQuery(operation="filter_count", narrative="conteo total")
    out = _execute(chat_df, query)
    assert out["table"].iloc[0]["valor"] == len(chat_df)


def test_value_counts_aggregates_by_column(chat_df):
    query = DataQuery(operation="value_counts", columns=["gender"], chart_type="bar", narrative="x")
    out = _execute(chat_df, query)
    assert set(out["table"]["gender"].tolist()) == {"F", "M"}
    assert out["chart"]["type"] == "bar"
    assert out["chart"]["x"] == "gender"


def test_groupby_count_with_two_keys(chat_df):
    query = DataQuery(
        operation="groupby_count",
        groupby=["gender", "region"],
        chart_type="bar",
        narrative="x",
    )
    out = _execute(chat_df, query)
    assert "conteo" in out["table"].columns
    assert out["chart"]["color"] == "region"


def test_groupby_agg_mean(chat_df):
    query = DataQuery(
        operation="groupby_agg",
        groupby=["region"],
        columns=["income"],
        agg="mean",
        chart_type="bar",
        narrative="x",
    )
    out = _execute(chat_df, query)
    assert "income" in out["table"].columns
    assert len(out["table"]) == chat_df["region"].nunique()


def test_distribution_returns_describe_and_histogram(chat_df):
    query = DataQuery(
        operation="distribution",
        columns=["age"],
        chart_type="histogram",
        narrative="x",
    )
    out = _execute(chat_df, query)
    assert "estadística" in out["table"].columns
    assert out["chart"]["type"] == "histogram"


def test_correlation_only_numeric(chat_df):
    query = DataQuery(
        operation="correlation",
        columns=["age", "income"],
        narrative="x",
    )
    out = _execute(chat_df, query)
    assert set(out["table"].columns) >= {"age", "income"}


def test_correlation_heatmap_builds_chart_with_matrix(chat_df):
    query = DataQuery(
        operation="correlation",
        columns=["age", "income"],
        chart_type="heatmap",
        narrative="x",
    )
    out = _execute(chat_df, query)
    assert out["chart"] is not None
    assert out["chart"]["type"] == "heatmap"
    matrix = out["chart"]["data"]
    # La matriz debe ser cuadrada (no la versión reset_index para la tabla)
    assert list(matrix.columns) == list(matrix.index)


def test_correlation_without_heatmap_chart_is_none(chat_df):
    query = DataQuery(
        operation="correlation",
        columns=["age", "income"],
        chart_type="table",
        narrative="x",
    )
    out = _execute(chat_df, query)
    assert out["chart"] is None


def test_groupby_count_normalize_row_pct_sums_100_per_group(chat_df):
    query = DataQuery(
        operation="groupby_count",
        groupby=["region", "gender"],
        normalize="row_pct",
        chart_type="bar",
        narrative="x",
    )
    out = _execute(chat_df, query)
    table = out["table"]
    assert "porcentaje" in table.columns
    assert "conteo" not in table.columns
    # Cada región debe sumar ~100 (puede haber rounding error muy pequeño).
    sums = table.groupby("region")["porcentaje"].sum()
    for s in sums:
        assert abs(s - 100.0) < 0.5
    # El chart debe apuntar a la columna renombrada.
    assert out["chart"]["y"] == "porcentaje"


def test_groupby_count_normalize_total_pct_sums_100_global(chat_df):
    query = DataQuery(
        operation="groupby_count",
        groupby=["region", "gender"],
        normalize="total_pct",
        chart_type="bar",
        narrative="x",
    )
    out = _execute(chat_df, query)
    table = out["table"]
    assert abs(table["porcentaje"].sum() - 100.0) < 0.5


def test_value_counts_normalize_total_pct(chat_df):
    query = DataQuery(
        operation="value_counts",
        columns=["gender"],
        normalize="total_pct",
        chart_type="bar",
        narrative="x",
    )
    out = _execute(chat_df, query)
    assert "porcentaje" in out["table"].columns
    assert abs(out["table"]["porcentaje"].sum() - 100.0) < 0.5


def test_needs_clarification_short_circuits_execution(chat_df):
    fake_query = DataQuery(
        operation="groupby_count",
        groupby=["region", "gender"],
        narrative="Tu pregunta puede leerse de varias formas.",
        needs_clarification=True,
        clarification_question="¿Cómo prefieres ver los resultados?",
        clarification_options=["Conteo absoluto", "% dentro de cada grupo", "% del total"],
    )
    p1, p2, p3 = _patch_llm(fake_query)
    with p1, p2, p3:
        result = answer_data_question(chat_df, "cuántos hombres por región", mode="raw")

    assert result.error is None
    assert result.table is None
    assert result.chart is None
    assert result.clarification is not None
    assert result.clarification["options"] == [
        "Conteo absoluto",
        "% dentro de cada grupo",
        "% del total",
    ]


def test_chart_type_accepts_line(chat_df):
    # `line` debe ser un valor válido del Literal y _execute debe propagarlo.
    query = DataQuery(
        operation="groupby_count",
        groupby=["region"],
        chart_type="line",
        narrative="x",
    )
    out = _execute(chat_df, query)
    assert out["chart"]["type"] == "line"


def test_top_n_categorical(chat_df):
    query = DataQuery(
        operation="top_n",
        columns=["region"],
        top_n=2,
        chart_type="bar",
        narrative="x",
    )
    out = _execute(chat_df, query)
    assert len(out["table"]) == 2


def test_validate_columns_catches_missing():
    query = DataQuery(operation="value_counts", columns=["does_not_exist"], narrative="x")
    err = _validate_columns(query, {"age", "gender"})
    assert err is not None and "does_not_exist" in err


def test_answer_data_question_full_pipeline(chat_df):
    fake_query = DataQuery(
        operation="value_counts",
        columns=["gender"],
        chart_type="bar",
        narrative="Cuento por género.",
    )
    p1, p2, p3 = _patch_llm(fake_query)
    with p1, p2, p3:
        result = answer_data_question(chat_df, "cuántos hombres y mujeres", context="", mode="raw")

    assert result.error is None
    assert result.narrative == "Cuento por género."
    assert result.table is not None and "conteo" in result.table.columns
    assert result.chart["type"] == "bar"


def test_answer_data_question_invalid_column_returns_error(chat_df):
    fake_query = DataQuery(
        operation="value_counts",
        columns=["nope"],
        chart_type="bar",
        narrative="x",
    )
    p1, p2, p3 = _patch_llm(fake_query)
    with p1, p2, p3:
        result = answer_data_question(chat_df, "?", mode="raw")

    assert result.error is not None
    assert result.table is None


def test_format_history_empty():
    assert "sin historial" in _format_history(None)
    assert "sin historial" in _format_history([])
    assert "sin historial" in _format_history([{"role": "user", "text": ""}])


def test_format_history_keeps_last_three_turns():
    history = [
        {"role": "user", "text": "p1"},
        {"role": "assistant", "text": "a1"},
        {"role": "user", "text": "p2"},
        {"role": "assistant", "text": "a2"},
        {"role": "user", "text": "p3"},
        {"role": "assistant", "text": "a3"},
        {"role": "user", "text": "p4"},
        {"role": "assistant", "text": "a4"},
    ]
    formatted = _format_history(history, max_turns=3)
    # Solo deben aparecer las últimas 3 vueltas (p2..a4)
    assert "p1" not in formatted
    assert "a1" not in formatted
    assert "p2" in formatted
    assert "Asistente: a4" in formatted


def test_format_history_truncates_long_messages():
    long = "x" * 500
    formatted = _format_history([{"role": "user", "text": long}])
    assert "…" in formatted
    # No debe superar 280 chars + prefijo + ellipsis
    assert len(formatted) < 320


def test_answer_data_question_passes_history_to_prompt(chat_df):
    fake_query = DataQuery(
        operation="filter_count",
        narrative="ok",
        chart_type="none",
    )
    captured = {}

    def fake_invoke(_llm, prompt, _schema, _fallback):
        captured["prompt"] = prompt
        return fake_query, None

    history = [
        {"role": "user", "text": "cuántos hombres mayores de 22"},
        {"role": "assistant", "text": "Hay 504 hombres mayores de 22 años."},
    ]

    with patch("src.llm.data_qa.invoke_json_with_retry", fake_invoke), \
         patch("src.llm.data_qa.get_llm_json", lambda: None), \
         patch("src.llm.data_qa._generate_natural_narrative", lambda *_a, **_kw: None):
        answer_data_question(chat_df, "y de esos cuántos viven en el norte", history=history)

    prompt = captured["prompt"]
    assert "Historial reciente" in prompt
    assert "cuántos hombres mayores de 22" in prompt
    assert "Hay 504 hombres mayores de 22" in prompt


def test_answer_data_question_without_history_uses_placeholder(chat_df):
    fake_query = DataQuery(operation="filter_count", narrative="ok", chart_type="none")
    captured = {}

    def fake_invoke(_llm, prompt, _schema, _fallback):
        captured["prompt"] = prompt
        return fake_query, None

    with patch("src.llm.data_qa.invoke_json_with_retry", fake_invoke), \
         patch("src.llm.data_qa.get_llm_json", lambda: None), \
         patch("src.llm.data_qa._generate_natural_narrative", lambda *_a, **_kw: None):
        answer_data_question(chat_df, "cuántas filas hay")

    assert "sin historial" in captured["prompt"]


def test_answer_data_question_llm_error_falls_through(chat_df):
    fake_query = DataQuery(
        operation="filter_count",
        narrative="No entendí, mostrando conteo.",
        chart_type="none",
    )
    p1, p2, p3 = _patch_llm(fake_query, error="boom")
    with p1, p2, p3:
        result = answer_data_question(chat_df, "?", mode="raw")

    assert result.error == "boom"
    assert "conteo" in result.narrative.lower() or "entendí" in result.narrative.lower()
