"""Tools DETERMINISTAS del chat agéntico.

El juicio vive en el agente; las tools son funciones puras sobre el DataFrame
(reutilizan el executor whitelisteado de data_qa). Ninguna tool llama a un LLM.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field

from src.llm.data_qa import _execute, _summarize_columns, _summarize_table_for_prompt
from src.models.schemas import DataQuery, FilterCondition, GroupFilterSpec


# --------------------------------------------------------------------------- #
# Schemas de tools (el docstring es la descripción que ve el modelo)
# --------------------------------------------------------------------------- #
class consultar_datos(DataQuery):
    """Ejecuta UNA operación de análisis sobre el dataset (determinista y segura).
    Operaciones: describe, value_counts, groupby_count, groupby_agg, distribution,
    correlation, top_n, filter_count, missing_values. Acepta filter_by, bins,
    normalize y chart_type. La tabla y gráfica de la ÚLTIMA llamada se muestran
    al usuario junto a tu respuesta final."""


class ver_esquema(BaseModel):
    """Devuelve las columnas del dataset con tipo, estadísticas básicas y valores
    frecuentes. Úsala si no estás seguro de qué columnas o valores existen."""


class ver_arquetipos(BaseModel):
    """Devuelve los arquetipos del análisis actual: nombre, tamaño, comportamiento
    principal y barreras. Solo disponible en modo arquetipos."""


class comparar_grupos(BaseModel):
    """Compara dos grupos lado a lado: tamaño, % del total, medias de columnas
    numéricas y categoría más frecuente de las categóricas. Define cada grupo con
    filtros (mismos operadores que filter_by de consultar_datos)."""
    label_a: str = Field(description="Nombre corto del grupo A para la tabla")
    filtros_a: List[FilterCondition]
    label_b: str = Field(description="Nombre corto del grupo B para la tabla")
    filtros_b: List[FilterCondition]
    columnas: Optional[List[str]] = Field(
        default=None, description="Columnas a comparar; por defecto las numéricas principales",
    )


TOOL_SCHEMAS = [consultar_datos, ver_esquema, ver_arquetipos, comparar_grupos]


# --------------------------------------------------------------------------- #
# Ejecución
# --------------------------------------------------------------------------- #
@dataclass
class ToolExecution:
    text: str
    table: Optional[pd.DataFrame] = None
    chart: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def _exec_consultar_datos(df: pd.DataFrame, args: Dict[str, Any]) -> ToolExecution:
    query = consultar_datos(**args)
    out = _execute(df, query)
    table = out.get("table")
    return ToolExecution(
        text=_summarize_table_for_prompt(table),
        table=table,
        chart=out.get("chart"),
    )


def _exec_ver_esquema(df: pd.DataFrame) -> ToolExecution:
    import json

    return ToolExecution(
        text=json.dumps(_summarize_columns(df), default=str, ensure_ascii=False),
    )


def _exec_ver_arquetipos(archetypes: Optional[List[Dict[str, Any]]]) -> ToolExecution:
    if not archetypes:
        return ToolExecution(text="No hay arquetipos disponibles en este modo (dataset sin analizar).")
    lines = []
    for a in archetypes:
        barreras = "; ".join((a.get("barreras") or [])[:2])
        lines.append(
            f"- {a.get('label')} (cluster {a.get('cluster_id')}, {a.get('size')} filas, "
            f"{a.get('prevalence')}%): {a.get('comportamiento_principal') or a.get('description', '')}"
            + (f" · Barreras: {barreras}" if barreras else "")
        )
    return ToolExecution(text="\n".join(lines))


def _exec_comparar_grupos(df: pd.DataFrame, args: Dict[str, Any]) -> ToolExecution:
    spec = comparar_grupos(**args)
    from src.llm.data_qa import _apply_filters

    group_a = _apply_filters(df, GroupFilterSpec(filter_by=spec.filtros_a))
    group_b = _apply_filters(df, GroupFilterSpec(filter_by=spec.filtros_b))
    label_a = spec.label_a.strip() or "Grupo A"
    label_b = spec.label_b.strip() or "Grupo B"
    if group_a.empty or group_b.empty:
        empty = label_a if group_a.empty else label_b
        return ToolExecution(
            text=f"El grupo '{empty}' quedó vacío con esos filtros — revisa columnas/valores con ver_esquema.",
            error="grupo vacío",
        )

    numeric_all = df.select_dtypes(include="number").columns.tolist()
    if spec.columnas:
        cols = [c for c in spec.columnas if c in df.columns]
    else:
        cols = numeric_all[:6]

    metricas: List[str] = ["filas", "% del total"]
    val_a: List[Any] = [len(group_a), round(len(group_a) / len(df) * 100, 1)]
    val_b: List[Any] = [len(group_b), round(len(group_b) / len(df) * 100, 1)]
    for col in cols:
        if col in numeric_all:
            metricas.append(f"{col} (promedio)")
            val_a.append(round(float(group_a[col].mean()), 2))
            val_b.append(round(float(group_b[col].mean()), 2))
        else:
            metricas.append(f"{col} (más frecuente)")
            top_a = group_a[col].mode()
            top_b = group_b[col].mode()
            val_a.append(str(top_a.iloc[0]) if not top_a.empty else "—")
            val_b.append(str(top_b.iloc[0]) if not top_b.empty else "—")

    table = pd.DataFrame({"métrica": metricas, label_a: val_a, label_b: val_b})
    return ToolExecution(text=table.to_string(index=False), table=table, chart=None)


def execute_tool(
    name: str,
    args: Dict[str, Any],
    df: pd.DataFrame,
    archetypes: Optional[List[Dict[str, Any]]] = None,
) -> ToolExecution:
    """Despacha una tool-call. Los errores vuelven como texto para que el agente corrija."""
    try:
        if name == "consultar_datos":
            return _exec_consultar_datos(df, args)
        if name == "ver_esquema":
            return _exec_ver_esquema(df)
        if name == "ver_arquetipos":
            return _exec_ver_arquetipos(archetypes)
        if name == "comparar_grupos":
            return _exec_comparar_grupos(df, args)
        return ToolExecution(text=f"Herramienta desconocida: {name}", error="unknown tool")
    except Exception as exc:  # noqa: BLE001 — el agente debe poder corregir y seguir
        return ToolExecution(text=f"Error al ejecutar {name}: {exc}", error=str(exc))
