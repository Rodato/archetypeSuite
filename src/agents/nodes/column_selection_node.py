"""Column selection node + helper.

The pipeline node respects an upstream `selected_columns` from state (set by UI
after user confirms) and otherwise asks the LLM for a recommendation.

`suggest_columns` is exposed for the UI to preview the selection before
running the pipeline.
"""
import json
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config.settings import settings
from src.data.column_filter import apply_static_filters
from src.llm.prompts import COLUMN_RELEVANCE_PROMPT
from src.llm.provider import get_llm_json, invoke_json_with_retry
from src.models.schemas import ColumnRecommendation, ColumnRelevanceDecision
from src.models.state import PipelineState


def _all_columns_recommendation(columns: List[str], reason: str) -> ColumnRelevanceDecision:
    return ColumnRelevanceDecision(
        selected_columns=[
            ColumnRecommendation(name=c, reason=reason, importance="medium") for c in columns
        ],
        excluded_columns=[],
        summary=reason,
    )


def _summarize_column(col: str, series: pd.Series) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "name": col,
        "dtype": str(series.dtype),
        "n_unique": int(series.nunique(dropna=True)),
        "pct_missing": round(float(series.isna().mean()), 3),
    }
    if pd.api.types.is_numeric_dtype(series):
        info["min"] = float(series.min()) if series.notna().any() else None
        info["max"] = float(series.max()) if series.notna().any() else None
        info["mean"] = float(series.mean()) if series.notna().any() else None
        if series.notna().any():
            info["sample"] = [
                float(x)
                for x in series.dropna()
                .sample(min(3, int(series.notna().sum())), random_state=settings.random_seed)
                .tolist()
            ]
    else:
        top = series.value_counts(dropna=True).head(5)
        info["top_values"] = {str(k): int(v) for k, v in top.items()}
    return info


def suggest_columns(
    df: pd.DataFrame,
    dataset_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Run static filters + LLM relevance suggestion.

    Returns a dict with:
      - filtered_df: DataFrame after static filters (datetimes already extracted)
      - static_filter_result: report from apply_static_filters
      - column_recommendation: dict of selected/excluded/summary from LLM
      - llm_error: error string if the LLM call failed (else None)
    """
    filtered_df, static_report = apply_static_filters(df)

    if filtered_df.empty or len(filtered_df.columns) == 0:
        return {
            "filtered_df": filtered_df,
            "static_filter_result": static_report,
            "column_recommendation": {
                "selected_columns": [],
                "excluded_columns": [],
                "summary": "No quedaron columnas tras filtros estáticos.",
            },
            "llm_error": None,
        }

    columns_summary = [_summarize_column(col, filtered_df[col]) for col in filtered_df.columns]
    context = dataset_context or "No se proporcionó contexto adicional."
    prompt = COLUMN_RELEVANCE_PROMPT.format(
        context=context,
        columns_summary=json.dumps(columns_summary, indent=2, default=str, ensure_ascii=False),
    )

    llm = get_llm_json()
    decision, error = invoke_json_with_retry(
        llm,
        prompt,
        ColumnRelevanceDecision,
        lambda: _all_columns_recommendation(
            list(filtered_df.columns),
            "LLM no disponible — se usan todas las columnas post-filtros.",
        ),
    )

    available = set(filtered_df.columns)
    valid_selected = [r for r in decision.selected_columns if r.name in available]
    if not valid_selected:
        decision = _all_columns_recommendation(
            list(filtered_df.columns),
            "Sugerencia del LLM no validó contra columnas disponibles — usando todas.",
        )
        valid_selected = list(decision.selected_columns)

    valid_excluded = [e for e in decision.excluded_columns if e.name in available]

    return {
        "filtered_df": filtered_df,
        "static_filter_result": static_report,
        "column_recommendation": {
            "selected_columns": [r.model_dump() for r in valid_selected],
            "excluded_columns": [e.model_dump() for e in valid_excluded],
            "summary": decision.summary,
        },
        "llm_error": error,
    }


def column_selection_node(state: PipelineState) -> dict:
    df = pd.DataFrame(state["raw_data"])
    logs: List[str] = []

    upstream_selection = state.get("selected_columns")
    upstream_static = state.get("static_filter_result")
    upstream_recommendation = state.get("column_recommendation")

    # Fast path: UI already ran suggest_columns + user confirmed.
    # We trust the upstream selection but still apply static filters defensively
    # in case the df came in unfiltered.
    if upstream_selection and upstream_static and upstream_recommendation:
        filtered_df, _ = apply_static_filters(df)
        valid = [c for c in upstream_selection if c in filtered_df.columns]
        if not valid:
            logs.append("[column_selection] selected_columns no validaron — usando todas las columnas post-filtros.")
            valid = list(filtered_df.columns)
        final_df = filtered_df[valid]
        logs.append(f"[column_selection] Usuario seleccionó {len(valid)} columnas: {', '.join(valid)}")
        return {
            "static_filter_result": upstream_static,
            "column_recommendation": upstream_recommendation,
            "selected_columns": valid,
            "datetime_columns": [d["original"] for d in upstream_static.get("datetime_extracted", [])],
            "raw_data": final_df.to_dict(orient="list"),
            "log_messages": logs,
        }

    # Slow path: run static filters + LLM suggestion, use LLM picks as final.
    result = suggest_columns(df, state.get("dataset_context"))
    static_report = result["static_filter_result"]
    recommendation = result["column_recommendation"]
    filtered_df = result["filtered_df"]

    if static_report["dropped"]:
        names = ", ".join(d["column"] for d in static_report["dropped"])
        logs.append(f"[column_selection] Filtros estáticos descartaron: {names}")
    if static_report["datetime_extracted"]:
        names = ", ".join(d["original"] for d in static_report["datetime_extracted"])
        logs.append(f"[column_selection] Fechas convertidas a año: {names}")
    if result.get("llm_error"):
        logs.insert(0, f"[column_selection] LLM falló, usando fallback. Error: {result['llm_error']}")

    selected_names = [r["name"] for r in recommendation["selected_columns"]]
    final_df = filtered_df[selected_names] if selected_names else filtered_df

    logs.append(
        f"[column_selection] LLM sugiere {len(selected_names)} columnas: "
        + (", ".join(selected_names) if selected_names else "ninguna")
    )

    return {
        "static_filter_result": static_report,
        "column_recommendation": recommendation,
        "selected_columns": selected_names,
        "datetime_columns": [d["original"] for d in static_report["datetime_extracted"]],
        "raw_data": final_df.to_dict(orient="list"),
        "log_messages": logs,
    }
