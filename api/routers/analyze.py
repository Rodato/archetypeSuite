"""Analyze endpoint: drives graph.stream and emits Server-Sent progress events."""
import json
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api.store import datasets, new_id, now_iso, save_run
from api.transform import build_run_record, run_summary
from src.agents.graph import compile_graph
from src.config.settings import settings
from src.data.column_filter import apply_static_filters
from src.core.quality import (
    NODE_FRIENDLY_MESSAGES,
    PIPELINE_UI_STEPS,
    nodes_with_logs,
)

router = APIRouter(prefix="/api/datasets", tags=["analyze"])

PIPELINE_KEYS = [k for k, _ in PIPELINE_UI_STEPS]
STEP_LABELS = dict(PIPELINE_UI_STEPS)


class AnalyzeBody(BaseModel):
    context: str = ""
    selected_columns: Optional[List[str]] = None
    static_filter_result: Optional[Dict[str, Any]] = None
    column_recommendation: Optional[Dict[str, Any]] = None


def _sse(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _progress_event(completed: set, running: Optional[str]) -> Dict[str, Any]:
    steps = []
    for key, label in PIPELINE_UI_STEPS:
        if key in completed:
            status = "done"
        elif key == running:
            status = "running"
        else:
            status = "pending"
        steps.append({"key": key, "label": label, "status": status})
    return {
        "type": "progress",
        "steps": steps,
        "running": running,
        "message": NODE_FRIENDLY_MESSAGES.get(running, "Procesando…") if running else "",
        "completed_count": len(completed),
        "total": len(PIPELINE_KEYS),
    }


@router.post("/{dataset_id}/analyze")
def analyze(dataset_id: str, body: AnalyzeBody) -> StreamingResponse:
    entry = datasets.get(dataset_id)
    if entry is None:
        raise HTTPException(404, "Dataset no encontrado o expirado. Vuelve a subir el archivo.")
    df = entry["df"]
    file_name = entry["file_name"]
    context = body.context or ""

    def event_stream():
        if not settings.openrouter_api_key:
            yield _sse({
                "type": "error",
                "error_type": "no_api_key",
                "message": "Falta configurar OPENROUTER_API_KEY en el backend (.env).",
            })
            return

        final_state: Optional[Dict[str, Any]] = None
        last_running: Optional[str] = None
        try:
            # Build initial state (resolve file_name locally to avoid global state races).
            df_for_pipeline = df
            fast_path = (
                body.selected_columns
                and body.static_filter_result is not None
                and body.column_recommendation is not None
            )
            if body.selected_columns:
                filtered_df, _ = apply_static_filters(df)
                cols = [c for c in body.selected_columns if c in filtered_df.columns]
                if cols:
                    df_for_pipeline = filtered_df[cols]

            initial_state: Dict[str, Any] = {
                "raw_data": df_for_pipeline.to_dict(orient="list"),
                "file_name": file_name,
                "dataset_context": context,
                "refinement_count": 0,
                "log_messages": [],
            }
            if fast_path:
                initial_state["selected_columns"] = body.selected_columns
                initial_state["static_filter_result"] = body.static_filter_result
                initial_state["column_recommendation"] = body.column_recommendation

            graph = compile_graph()
            last_running = PIPELINE_KEYS[0]
            yield _sse(_progress_event(set(), PIPELINE_KEYS[0]))

            for state in graph.stream(initial_state, stream_mode="values"):
                final_state = state
                logs = state.get("log_messages", [])
                completed = nodes_with_logs(logs) & set(PIPELINE_KEYS)
                running = next((k for k in PIPELINE_KEYS if k not in completed), PIPELINE_KEYS[-1])
                last_running = running
                yield _sse(_progress_event(completed, running))

            if final_state is None:
                yield _sse({"type": "error", "error_type": "empty", "message": "No se generó resultado."})
                return

            run_id = new_id()
            record = build_run_record(
                final_state,
                run_id=run_id,
                created_at=now_iso(),
                file_name=file_name,
                dataset_context=context,
            )
            save_run(record)
            yield _sse({"type": "done", "run_id": run_id, "summary": run_summary(record)})
        except Exception as exc:  # noqa: BLE001 — surface any failure as a clean SSE error event
            yield _sse({
                "type": "error",
                "error_type": type(exc).__name__,
                "message": f"El análisis se interrumpió: {exc}",
                "failed_step": last_running,
            })
            return

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
