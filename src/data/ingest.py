from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd
import sqlalchemy

# Extended missing-value tokens: pandas' defaults + common Spanish/LatAm sentinels.
# Without this, a column full of "-" / "sin dato" / "?" reads as text and the app
# wrongly reports "no missing values".
NA_TOKENS = set(pd.io.parsers.readers.STR_NA_VALUES) | {
    "?", "-", "--", ".", "..", "none", "None", "NONE",
    "n/a", "N/A", "#N/A", "na", "n.a.", "n. a.",
    "n/d", "n.d.", "s/d", "s.d.", "sin dato", "Sin dato", "SIN DATO",
    "sin datos", "no aplica", "No aplica", "NO APLICA",
    "missing", "Missing", "MISSING", "(blank)", "(vacío)", "vacío", "nulo", "Nulo",
}

_CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")


def coerce_numeric_columns(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Salvage numeric columns read as text (stray sentinels, thousands separators, $).

    For each object column, if >= `threshold` of its non-null values parse as numbers
    after stripping thousands separators / currency symbols, convert it to numeric.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        s = df[col]
        non_null = s.dropna()
        if non_null.empty:
            continue
        cleaned = (
            non_null.astype(str)
            .str.strip()
            .str.replace(r"[,\s]", "", regex=True)
            .str.replace(r"^[$€£]\s?", "", regex=True)
            .str.replace("%", "", regex=False)
        )
        coerced = pd.to_numeric(cleaned, errors="coerce")
        share = coerced.notna().mean()
        # Require a real numeric column (not a 2-value coded category masquerading as numeric).
        if share >= threshold and non_null.nunique() > 2:
            full = (
                s.astype(str).str.strip()
                .str.replace(r"[,\s]", "", regex=True)
                .str.replace(r"^[$€£]\s?", "", regex=True)
                .str.replace("%", "", regex=False)
            )
            df[col] = pd.to_numeric(full, errors="coerce")
    return df


def read_csv_bytes(content: bytes) -> pd.DataFrame:
    """Read CSV bytes trying several encodings (LatAm Excel exports are often cp1252)."""
    last_err: Optional[Exception] = None
    for enc in _CSV_ENCODINGS:
        try:
            return pd.read_csv(
                io.BytesIO(content), encoding=enc, na_values=list(NA_TOKENS),
                keep_default_na=True, skipinitialspace=True,
            )
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            last_err = exc
            continue
    if last_err:
        raise last_err
    raise pd.errors.ParserError("No se pudo leer el CSV.")


def read_excel_bytes(content: bytes, sheet: Optional[str] = None) -> Tuple[pd.DataFrame, list, str]:
    """Read an Excel file. Returns (df, all_sheet_names, used_sheet).

    If no sheet is given, auto-picks the sheet with the most non-empty cells (instead of
    blindly using sheet 0, which silently analyzed the wrong sheet).
    """
    xls = pd.ExcelFile(io.BytesIO(content))
    sheets = list(xls.sheet_names)
    if sheet and sheet in sheets:
        used = sheet
    else:
        # pick the richest sheet
        best, best_cells = sheets[0], -1
        for name in sheets:
            preview = xls.parse(name, na_values=list(NA_TOKENS), keep_default_na=True)
            cells = int(preview.notna().to_numpy().sum())
            if cells > best_cells:
                best, best_cells = name, cells
        used = best
    df = xls.parse(used, na_values=list(NA_TOKENS), keep_default_na=True)
    return df, sheets, used


def read_upload(filename: str, content: bytes, sheet: Optional[str] = None) -> Tuple[pd.DataFrame, dict]:
    """Read an uploaded CSV/Excel robustly. Returns (df, meta) where meta carries sheet info."""
    name = (filename or "").lower()
    meta: dict = {}
    if name.endswith(".csv"):
        df = read_csv_bytes(content)
    elif name.endswith((".xlsx", ".xls")):
        df, sheets, used = read_excel_bytes(content, sheet)
        meta = {"sheets": sheets, "sheet": used}
    else:
        raise ValueError("Formato no soportado. Sube un CSV o Excel (.xlsx/.xls).")
    df = coerce_numeric_columns(df)
    return df, meta


class DataIngestor:

    def load(self, source: Union[str, pd.DataFrame]) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return source

        if isinstance(source, str):
            ext = Path(source).suffix.lower()
            if ext == ".csv":
                with open(source, "rb") as f:
                    df = read_csv_bytes(f.read())
            elif ext in (".xlsx", ".xls"):
                with open(source, "rb") as f:
                    df, _, _ = read_excel_bytes(f.read())
            else:
                raise ValueError(f"Unsupported file extension: {ext}")
            return coerce_numeric_columns(df)

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
