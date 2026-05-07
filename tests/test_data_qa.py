from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.llm.data_qa import _execute, _validate_columns, answer_data_question
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
