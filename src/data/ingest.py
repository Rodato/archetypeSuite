from __future__ import annotations

from pathlib import Path

import pandas as pd
import sqlalchemy


class DataIngestor:

    def load(self, source: str | pd.DataFrame) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return source

        if isinstance(source, str):
            ext = Path(source).suffix.lower()
            if ext == ".csv":
                return pd.read_csv(source)
            elif ext in (".xlsx", ".xls"):
                return pd.read_excel(source)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

        raise ValueError(f"Unsupported source type: {type(source).__name__}")

    def validate(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("El archivo está vacío.")
        if df.shape[1] < 2:
            raise ValueError(
                f"Necesitamos al menos 2 variables (columnas) para segmentar; el archivo trae {df.shape[1]}."
            )
        if df.shape[0] < 2:
            raise ValueError(
                f"Necesitamos al menos 2 filas para encontrar patrones; el archivo trae {df.shape[0]}."
            )

    def load_sql(self, connection_string: str, query: str) -> pd.DataFrame:
        engine = sqlalchemy.create_engine(connection_string)
        return pd.read_sql(query, engine)
