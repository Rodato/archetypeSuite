"""Conversational data Q&A: LLM picks an operation, Python executes it."""
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from src.llm.prompts import DATA_QA_PROMPT
from src.llm.provider import get_llm_json, invoke_json_with_retry
from src.models.schemas import DataQuery


MODE_DESCRIPTIONS = {
    "raw": (
        "El dataset es el dataset crudo subido por el usuario. "
        "Las preguntas suelen ser exploratorias: distribuciones, conteos, correlaciones."
    ),
    "archetypes": (
        "El dataset incluye los datos originales más dos columnas extra: "
        "`Cluster` (entero) y `Arquetipo` (etiqueta del arquetipo). "
        "Las preguntas son sobre comparar arquetipos entre sí."
    ),
}


@dataclass
class DataQAResult:
    narrative: str
    operation: str
    table: Optional[pd.DataFrame] = None
    chart: Optional[Dict[str, Any]] = None  # {type, data, x, y, color}
    error: Optional[str] = None


def _summarize_columns(df: pd.DataFrame, max_cols: int = 30) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for col in list(df.columns)[:max_cols]:
        s = df[col]
        info: Dict[str, Any] = {"name": col, "dtype": str(s.dtype)}
        if pd.api.types.is_numeric_dtype(s):
            info["min"] = float(s.min()) if s.notna().any() else None
            info["max"] = float(s.max()) if s.notna().any() else None
            info["sample"] = s.dropna().head(3).tolist()
        else:
            top = s.value_counts(dropna=True).head(5)
            info["top_values"] = {str(k): int(v) for k, v in top.items()}
        summary.append(info)
    return summary


def _validate_columns(query: DataQuery, available: set) -> Optional[str]:
    for col in (query.columns or []):
        if col not in available:
            return f"La columna `{col}` no existe en el dataset."
    for col in (query.groupby or []):
        if col not in available:
            return f"La columna de agrupación `{col}` no existe."
    return None


def _execute(df: pd.DataFrame, query: DataQuery) -> Dict[str, Any]:
    """Run the operation deterministically. Returns dict with 'table' and optional 'chart'."""
    op = query.operation
    if op == "filter_count":
        table = pd.DataFrame({"métrica": ["total_filas"], "valor": [len(df)]})
        return {"table": table, "chart": None}

    if op == "describe":
        cols = query.columns or df.select_dtypes(include="number").columns.tolist()
        cols = [c for c in cols if c in df.columns]
        table = df[cols].describe().reset_index().rename(columns={"index": "estadística"})
        return {"table": table, "chart": None}

    if op == "value_counts":
        if not query.columns:
            raise ValueError("value_counts requiere una columna.")
        col = query.columns[0]
        counts = df[col].value_counts(dropna=False).reset_index()
        counts.columns = [col, "conteo"]
        chart = {"type": query.chart_type, "data": counts, "x": col, "y": "conteo", "color": None}
        return {"table": counts, "chart": chart}

    if op == "groupby_count":
        if not query.groupby:
            raise ValueError("groupby_count requiere groupby.")
        gb = df.groupby(query.groupby, dropna=False).size().reset_index(name="conteo")
        chart = {
            "type": query.chart_type,
            "data": gb,
            "x": query.groupby[0],
            "y": "conteo",
            "color": query.groupby[1] if len(query.groupby) > 1 else None,
        }
        return {"table": gb, "chart": chart}

    if op == "groupby_agg":
        if not query.groupby or not query.columns:
            raise ValueError("groupby_agg requiere groupby y columns.")
        agg = query.agg or "mean"
        target = query.columns[0]
        gb = df.groupby(query.groupby, dropna=False)[target].agg(agg).reset_index()
        chart = {
            "type": query.chart_type,
            "data": gb,
            "x": query.groupby[0],
            "y": target,
            "color": query.groupby[1] if len(query.groupby) > 1 else None,
        }
        return {"table": gb, "chart": chart}

    if op == "distribution":
        if not query.columns:
            raise ValueError("distribution requiere una columna.")
        col = query.columns[0]
        table = df[[col]].describe().reset_index().rename(columns={"index": "estadística"})
        chart = {
            "type": query.chart_type if query.chart_type != "table" else "histogram",
            "data": df[[col]].dropna(),
            "x": col,
            "y": None,
            "color": None,
        }
        return {"table": table, "chart": chart}

    if op == "correlation":
        cols = [c for c in (query.columns or []) if c in df.columns]
        if not cols:
            cols = df.select_dtypes(include="number").columns.tolist()
        corr = df[cols].corr().round(3).reset_index().rename(columns={"index": "variable"})
        return {"table": corr, "chart": None}

    if op == "top_n":
        if not query.columns:
            raise ValueError("top_n requiere una columna.")
        col = query.columns[0]
        n = query.top_n or 10
        if pd.api.types.is_numeric_dtype(df[col]):
            top = df.nlargest(n, col)[[col]]
        else:
            top = df[col].value_counts().head(n).reset_index()
            top.columns = [col, "conteo"]
        chart_y = "conteo" if "conteo" in top.columns else col
        chart = {"type": query.chart_type, "data": top, "x": col, "y": chart_y, "color": None}
        return {"table": top, "chart": chart}

    raise ValueError(f"Operación no soportada: {op}")


def answer_data_question(
    df: pd.DataFrame,
    question: str,
    *,
    context: str = "",
    mode: str = "raw",
) -> DataQAResult:
    columns_summary = _summarize_columns(df)
    prompt = DATA_QA_PROMPT.format(
        mode=mode,
        mode_description=MODE_DESCRIPTIONS.get(mode, MODE_DESCRIPTIONS["raw"]),
        context=context or "No se proporcionó contexto adicional.",
        columns_summary=json.dumps(columns_summary, indent=2, default=str, ensure_ascii=False),
        question=question,
    )

    llm = get_llm_json()
    fallback = lambda: DataQuery(
        operation="filter_count",
        narrative="No pude entender la pregunta — te muestro el conteo total de filas.",
        chart_type="none",
    )
    query, error = invoke_json_with_retry(llm, prompt, DataQuery, fallback)

    if error:
        return DataQAResult(
            narrative=query.narrative or "No pude responder la pregunta.",
            operation=query.operation,
            error=error,
        )

    validation_error = _validate_columns(query, set(df.columns))
    if validation_error:
        return DataQAResult(
            narrative=f"{query.narrative} (Pero {validation_error})",
            operation=query.operation,
            error=validation_error,
        )

    try:
        execution = _execute(df, query)
    except Exception as exc:
        return DataQAResult(
            narrative=f"No pude ejecutar la operación: {exc}",
            operation=query.operation,
            error=str(exc),
        )

    return DataQAResult(
        narrative=query.narrative,
        operation=query.operation,
        table=execution.get("table"),
        chart=execution.get("chart"),
    )
