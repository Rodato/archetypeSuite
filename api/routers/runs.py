"""Run endpoints: list / get / archetypes-mode chat / exports / delete."""
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from api.store import delete_run, get_run, list_runs
from api.transform import label_map_from_archetypes, serialize_qa_result
from src.llm.data_qa import answer_data_question
from src.core.export import archetypes_to_csv, build_markdown_report, labels_to_csv

router = APIRouter(prefix="/api/runs", tags=["runs"])


class ChatBody(BaseModel):
    question: str
    context: str = ""
    history: Optional[List[Dict[str, str]]] = None


@router.get("")
def all_runs() -> Dict[str, Any]:
    from api.transform import run_summary

    summaries = []
    for r in list_runs():
        try:
            summaries.append(run_summary(r))
        except Exception:  # noqa: BLE001 — a single bad file must not break the list
            continue
    return {"runs": summaries}


def _require_run(run_id: str) -> Dict[str, Any]:
    record = get_run(run_id)
    if record is None:
        raise HTTPException(404, "Análisis no encontrado.")
    return record


@router.get("/{run_id}")
def one_run(run_id: str) -> Dict[str, Any]:
    return _require_run(run_id)


@router.delete("/{run_id}")
def remove_run(run_id: str) -> Dict[str, Any]:
    if not delete_run(run_id):
        raise HTTPException(404, "Análisis no encontrado.")
    return {"ok": True}


def _reconstruct_labeled_df(record: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame(record.get("raw_data", {}))
    labels = record.get("labels", [])
    if len(labels) == len(df):
        df["Cluster"] = labels
        label_map = label_map_from_archetypes(record.get("archetypes", []))
        df["Arquetipo"] = df["Cluster"].map(label_map).fillna("Desconocido")
    return df


@router.post("/{run_id}/chat")
def chat(run_id: str, body: ChatBody) -> Dict[str, Any]:
    record = _require_run(run_id)
    df = _reconstruct_labeled_df(record)
    result = answer_data_question(
        df, body.question,
        context=body.context or record.get("dataset_context", ""),
        mode="archetypes", history=body.history,
    )
    return serialize_qa_result(result)


def _result_like(record: Dict[str, Any]) -> Dict[str, Any]:
    metrics = {k: v for k, v in (record.get("metrics") or {}).items() if isinstance(v, (int, float))}
    return {
        "file_name": record.get("file_name", "desconocido"),
        "dataset_context": record.get("dataset_context", ""),
        "selected_algorithm": record.get("advanced", {}).get("selected_algorithm", "KMeans"),
        "n_clusters": record.get("n_clusters", "N/A"),
        "refinement_count": record.get("advanced", {}).get("refinement_count", 0),
        "metrics": metrics,
        "archetypes": record.get("archetypes", []),
    }


@router.get("/{run_id}/export/archetypes.csv")
def export_archetypes(run_id: str) -> Response:
    record = _require_run(run_id)
    data = archetypes_to_csv(record.get("archetypes", []))
    return Response(
        content=data, media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=arquetipos.csv"},
    )


@router.get("/{run_id}/export/labeled.csv")
def export_labeled(run_id: str) -> Response:
    record = _require_run(run_id)
    data = labels_to_csv(
        record.get("raw_data", {}), record.get("labels", []), record.get("archetypes", []),
    )
    return Response(
        content=data, media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=datos_etiquetados.csv"},
    )


@router.get("/{run_id}/export/report.md")
def export_report(run_id: str) -> Response:
    record = _require_run(run_id)
    md = build_markdown_report(_result_like(record))
    return Response(
        content=md.encode("utf-8"), media_type="text/markdown",
        headers={"Content-Disposition": "attachment; filename=reporte_arquetipos.md"},
    )
