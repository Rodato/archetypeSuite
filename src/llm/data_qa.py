"""Conversational data Q&A: LLM picks an operation, Python executes it."""
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from src.llm.prompts import DATA_QA_PROMPT, NATURAL_ANSWER_PROMPT
from src.llm.provider import get_fast_text_llm, get_llm_json, invoke_json_with_retry
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
    clarification: Optional[Dict[str, Any]] = None  # {question, options}


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
    for f in (query.filter_by or []):
        if f.column not in available:
            return f"La columna de filtro `{f.column}` no existe."
    return None


def _apply_filters(df: pd.DataFrame, query: DataQuery) -> pd.DataFrame:
    if not query.filter_by:
        return df
    for cond in query.filter_by:
        col = cond.column
        if col not in df.columns:
            continue
        val = cond.value
        s = df[col]
        is_str = s.dtype == object or pd.api.types.is_string_dtype(s)
        # Numeric column compared against a string-typed value → coerce, else the
        # comparison raises an opaque TypeError. Skip the condition if uncoercible.
        if pd.api.types.is_numeric_dtype(s) and cond.op in ("gt", "lt", "gte", "lte", "eq", "ne"):
            coerced = pd.to_numeric(val, errors="coerce")
            if pd.isna(coerced):
                continue
            val = coerced
        if cond.op == "eq":
            mask = s.str.lower() == str(val).lower() if is_str else s == val
        elif cond.op == "ne":
            mask = s.str.lower() != str(val).lower() if is_str else s != val
        elif cond.op == "gt":
            mask = s > val
        elif cond.op == "lt":
            mask = s < val
        elif cond.op == "gte":
            mask = s >= val
        elif cond.op == "lte":
            mask = s <= val
        elif cond.op == "in":
            vals = val if isinstance(val, list) else [val]
            if is_str:
                lower_vals = [str(v).lower() for v in vals]
                mask = s.str.lower().isin(lower_vals)
            else:
                mask = s.isin(vals)
        elif cond.op == "contains":
            mask = s.astype(str).str.contains(str(val), case=False, na=False)
        else:
            continue
        df = df[mask]
    return df


def _apply_bins(df: pd.DataFrame, query: DataQuery) -> pd.DataFrame:
    """Replace columns referenced in `query.bins` with binned categorical versions."""
    if not query.bins:
        return df
    df = df.copy()
    for spec in query.bins:
        if spec.column not in df.columns or len(spec.edges) < 2:
            continue
        labels = spec.labels
        if labels is not None and len(labels) != len(spec.edges) - 1:
            labels = None
        df[spec.column] = pd.cut(
            df[spec.column],
            bins=spec.edges,
            labels=labels,
            include_lowest=True,
            right=True,
        ).astype(object)
    return df


def _apply_normalize(
    table: pd.DataFrame,
    *,
    value_col: str,
    primary: Optional[str],
    mode: str,
) -> str:
    """Convert a count column into a percentage in-place. Returns the new column name.

    - mode="row_pct" + `primary` set: % dentro de cada valor de `primary` (suma 100% por grupo).
      Si no hay `primary`, cae a total_pct (un solo nivel = mismo resultado).
    - mode="total_pct": % sobre el total general.
    - mode="none" (o cualquier otro): no toca la tabla.
    """
    if mode == "none" or value_col not in table.columns:
        return value_col
    if mode == "row_pct" and primary and primary in table.columns:
        totals = table.groupby(primary)[value_col].transform("sum")
        table[value_col] = (table[value_col] / totals * 100).round(1)
    else:
        total = table[value_col].sum()
        if total > 0:
            table[value_col] = (table[value_col] / total * 100).round(1)
    table.rename(columns={value_col: "porcentaje"}, inplace=True)
    return "porcentaje"


def _execute(df: pd.DataFrame, query: DataQuery) -> Dict[str, Any]:
    """Run the operation deterministically. Returns dict with 'table' and optional 'chart'."""
    df = _apply_bins(df, query)
    df = _apply_filters(df, query)
    op = query.operation
    if op == "filter_count":
        table = pd.DataFrame({"métrica": ["filas encontradas"], "valor": [len(df)]})
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
        value_col = _apply_normalize(counts, value_col="conteo", primary=None, mode=query.normalize)
        chart = {"type": query.chart_type, "data": counts, "x": col, "y": value_col, "color": None}
        return {"table": counts, "chart": chart}

    if op == "groupby_count":
        if not query.groupby:
            raise ValueError("groupby_count requiere groupby.")
        gb = df.groupby(query.groupby, dropna=False).size().reset_index(name="conteo")
        primary = query.groupby[0] if len(query.groupby) >= 2 else None
        value_col = _apply_normalize(gb, value_col="conteo", primary=primary, mode=query.normalize)
        chart = {
            "type": query.chart_type,
            "data": gb,
            "x": query.groupby[0],
            "y": value_col,
            "color": query.groupby[1] if len(query.groupby) > 1 else None,
        }
        return {"table": gb, "chart": chart}

    if op == "groupby_agg":
        if not query.groupby or not query.columns:
            raise ValueError("groupby_agg requiere groupby y columns.")
        agg = query.agg or "mean"
        targets = [c for c in query.columns if c in df.columns]
        if not targets:
            raise ValueError("Ninguna columna objetivo es válida.")
        gb = df.groupby(query.groupby, dropna=False)[targets].agg(agg).reset_index()
        if len(targets) > 1:
            numeric_cols = [c for c in gb.columns if c in targets]
            gb[numeric_cols] = gb[numeric_cols].round(3)
        chart_y = targets[0] if len(targets) == 1 else None
        chart = {
            "type": query.chart_type if len(targets) == 1 else "table",
            "data": gb,
            "x": query.groupby[0],
            "y": chart_y,
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
        corr_matrix = df[cols].corr().round(3)
        table = corr_matrix.reset_index().rename(columns={"index": "variable"})
        chart = None
        if query.chart_type == "heatmap" and not corr_matrix.empty:
            chart = {
                "type": "heatmap",
                "data": corr_matrix,
                "x": None,
                "y": None,
                "color": None,
            }
        return {"table": table, "chart": chart}

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

    if op == "missing_values":
        n_rows = len(df)
        miss = df.isna().sum()
        per_col = pd.DataFrame({
            "columna": list(miss.index),
            "faltantes": [int(v) for v in miss.values],
            "% faltantes": [round(float(v) / n_rows * 100, 1) if n_rows else 0.0 for v in miss.values],
        })
        per_col = per_col[per_col["faltantes"] > 0].sort_values("faltantes", ascending=False).reset_index(drop=True)
        if per_col.empty:
            table = pd.DataFrame({"métrica": ["valores faltantes"], "valor": [0]})
            return {"table": table, "chart": None}
        # A bar of missing-per-column is the useful default (ignore "table"/other inertia).
        chart_type = "none" if query.chart_type == "none" else "bar"
        chart = {"type": chart_type, "data": per_col, "x": "columna", "y": "faltantes", "color": None}
        return {"table": per_col, "chart": chart}

    raise ValueError(f"Operación no soportada: {op}")


def _summarize_table_for_prompt(table: pd.DataFrame, max_rows: int = 12) -> str:
    if table is None or table.empty:
        return "(sin resultados)"
    if len(table) > max_rows:
        head = table.head(max_rows).to_string(index=False)
        return f"{head}\n... ({len(table) - max_rows} filas más)"
    return table.to_string(index=False)


def _format_history(history: Optional[List[Dict[str, str]]], max_turns: int = 3) -> str:
    """Format the last N user/assistant turns for the prompt.

    Each item in `history` is `{"role": "user" | "assistant", "text": str}`.
    Returns "(sin historial)" when empty so the prompt block stays legible.
    """
    if not history:
        return "(sin historial — esta es la primera pregunta)"
    cleaned = [h for h in history if h.get("text")]
    if not cleaned:
        return "(sin historial — esta es la primera pregunta)"
    # Cada vuelta = 1 user + 1 assistant ⇒ tomamos las últimas 2*N entradas.
    tail = cleaned[-(max_turns * 2):]
    lines = []
    for entry in tail:
        label = "Usuario" if entry["role"] == "user" else "Asistente"
        text = entry["text"].strip().replace("\n", " ")
        if len(text) > 280:
            text = text[:277] + "…"
        lines.append(f"{label}: {text}")
    return "\n".join(lines)


def _generate_natural_narrative(question: str, table: pd.DataFrame, operation: str = "") -> Optional[str]:
    """Second LLM call: turn a structured result into a conversational answer."""
    try:
        prompt = NATURAL_ANSWER_PROMPT.format(
            question=question,
            operation=operation or "(desconocida)",
            result=_summarize_table_for_prompt(table),
        )
        llm = get_fast_text_llm()
        response = llm.invoke(prompt)
        text = (response.content or "").strip()
        return text or None
    except Exception:
        return None


def answer_data_question(
    df: pd.DataFrame,
    question: str,
    *,
    context: str = "",
    mode: str = "raw",
    history: Optional[List[Dict[str, str]]] = None,
) -> DataQAResult:
    columns_summary = _summarize_columns(df)
    prompt = DATA_QA_PROMPT.format(
        mode=mode,
        mode_description=MODE_DESCRIPTIONS.get(mode, MODE_DESCRIPTIONS["raw"]),
        context=context or "No se proporcionó contexto adicional.",
        columns_summary=json.dumps(columns_summary, indent=2, default=str, ensure_ascii=False),
        history=_format_history(history),
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

    # Cortocircuito: el LLM detectó ambigüedad (absoluto vs relativo, etc.).
    # No ejecutamos — devolvemos la pregunta para que la UI muestre chips.
    if query.needs_clarification and query.clarification_question and query.clarification_options:
        return DataQAResult(
            narrative=query.narrative or "Antes de calcular, ¿en qué formato lo quieres?",
            operation=query.operation,
            clarification={
                "question": query.clarification_question,
                "options": list(query.clarification_options),
            },
        )

    try:
        execution = _execute(df, query)
    except Exception as exc:
        return DataQAResult(
            narrative=f"No pude ejecutar la operación: {exc}",
            operation=query.operation,
            error=str(exc),
        )

    table = execution.get("table")
    natural = _generate_natural_narrative(question, table, query.operation)
    return DataQAResult(
        narrative=natural or query.narrative,
        operation=query.operation,
        table=table,
        chart=execution.get("chart"),
    )
