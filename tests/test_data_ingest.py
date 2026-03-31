import pandas as pd
import pytest

from src.data.ingest import DataIngestor


class TestDataIngestor:
    def test_load_dataframe(self, sample_df):
        ingestor = DataIngestor()
        result = ingestor.load(sample_df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == sample_df.shape

    def test_load_csv(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "test.csv"
        df.to_csv(path, index=False)
        ingestor = DataIngestor()
        result = ingestor.load(str(path))
        assert result.shape == (2, 2)

    def test_load_excel(self, tmp_path):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "test.xlsx"
        df.to_excel(path, index=False)
        ingestor = DataIngestor()
        result = ingestor.load(str(path))
        assert result.shape == (2, 2)

    def test_load_unsupported_extension(self):
        ingestor = DataIngestor()
        with pytest.raises(ValueError, match="Unsupported file extension"):
            ingestor.load("test.json")

    def test_load_unsupported_type(self):
        ingestor = DataIngestor()
        with pytest.raises(ValueError, match="Unsupported source type"):
            ingestor.load(123)

    def test_validate_empty(self):
        ingestor = DataIngestor()
        with pytest.raises(ValueError, match="empty"):
            ingestor.validate(pd.DataFrame())

    def test_validate_too_few_columns(self):
        ingestor = DataIngestor()
        with pytest.raises(ValueError, match="at least 2 columns"):
            ingestor.validate(pd.DataFrame({"a": range(20)}))

    def test_validate_too_few_rows(self):
        ingestor = DataIngestor()
        with pytest.raises(ValueError, match="at least 10 rows"):
            ingestor.validate(pd.DataFrame({"a": range(5), "b": range(5)}))

    def test_validate_success(self, sample_df):
        ingestor = DataIngestor()
        ingestor.validate(sample_df)  # should not raise
