"""Tests for ingestion robustness + analysis coherence fixes."""
import pandas as pd

from src.data.ingest import coerce_numeric_columns, read_upload
from src.llm.data_qa import _apply_filters, _execute
from src.models.schemas import DataQuery, FilterCondition


class TestIngestionRobustness:
    def test_sentinels_detected_as_missing(self):
        csv = b"a,b\n1,x\n2,-\n3,sin dato\n4,y"
        df, _ = read_upload("t.csv", csv)
        # "-" and "sin dato" become NaN, not literal strings
        assert int(df["b"].isna().sum()) == 2

    def test_numeric_with_thousands_and_sentinel_coerced(self):
        # Sentinels are already NaN after read (na_values); coerce salvages the numeric col.
        df = pd.DataFrame({"income": ['32,000', '54000', None, '1,200', '9000'], "g": list("MFMFM")})
        out = coerce_numeric_columns(df)
        assert pd.api.types.is_numeric_dtype(out["income"])
        assert int(out["income"].isna().sum()) == 1
        assert out["g"].tolist() == list("MFMFM")  # small category untouched

    def test_small_binary_category_not_coerced(self):
        # all-numeric but only 2 distinct → a coded category, keep as-is
        df = pd.DataFrame({"flag": ["0", "1", "1", "0", "1"]})
        out = coerce_numeric_columns(df)
        assert not pd.api.types.is_numeric_dtype(out["flag"])


class TestMissingValuesOperation:
    def test_reports_missing_per_column(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", None], "c": [1, 2, 3]})
        result = _execute(df, DataQuery(operation="missing_values"))
        table = result["table"]
        # only columns with missing appear
        assert set(table["columna"]) == {"a", "b"}
        assert result["chart"]["type"] == "bar"

    def test_no_missing_returns_zero(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = _execute(df, DataQuery(operation="missing_values"))
        assert result["table"]["valor"].iloc[0] == 0
        assert result["chart"] is None


class TestNumericFilterCoercion:
    def test_string_value_on_numeric_column_does_not_crash(self):
        df = pd.DataFrame({"age": [20, 30, 40, 50]})
        # value comes as a string "25" — must coerce, not TypeError
        q = DataQuery(operation="filter_count", filter_by=[FilterCondition(column="age", op="gt", value="25")])
        out = _apply_filters(df, q)
        assert len(out) == 3  # 30, 40, 50

    def test_uncoercible_value_skips_condition(self):
        df = pd.DataFrame({"age": [20, 30, 40]})
        q = DataQuery(operation="filter_count", filter_by=[FilterCondition(column="age", op="gt", value="abc")])
        out = _apply_filters(df, q)
        assert len(out) == 3  # condition skipped, nothing filtered
