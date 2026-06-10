"""Dataset endpoints: upload / sample / profile / column suggestion / raw-mode chat."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from api.serialization import to_jsonable
from api.store import datasets
from api.transform import classify_columns, preview_rows, serialize_qa_result
from src.agents.nodes.column_selection_node import suggest_columns
from src.data.ingest import DataIngestor, read_upload
from src.data.profiler import DataProfiler
from src.llm.data_qa import answer_data_question

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

# Demo: bienestar digital (900×14, generado con estructura comportamental plantada —
# ver sample_data/generate_bienestar_digital.py). El CSV "social_media_user_behavior.csv"
# original es sintético-uniforme (sin clusters reales, silhouette ~0.07) y queda solo
# para pruebas drag&drop.
SAMPLE_PATH = Path(__file__).resolve().parents[2] / "sample_data" / "bienestar_digital.csv"

# Friendly Spanish messages for common parse failures (ported from datos.LOAD_ERROR_MAP).
_LOAD_ERROR_MAP = {
    "ParserError": "No pudimos leer el archivo. Si es CSV, revisa el separador (puede usar ; o tab).",
    "UnicodeDecodeError": "El archivo usa una codificación que no reconocemos. Guárdalo como UTF-8.",
    "EmptyDataError": "El archivo parece estar vacío.",
}


class ContextBody(BaseModel):
    context: str = ""


class ChatBody(BaseModel):
    question: str
    context: str = ""
    history: Optional[List[Dict[str, str]]] = None


def _read_upload(filename: str, content: bytes) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    try:
        return read_upload(filename or "archivo", content)
    except ValueError as exc:  # unsupported format etc. — message is user-facing
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 — surface a friendly message
        msg = _LOAD_ERROR_MAP.get(type(exc).__name__, "No pudimos leer el archivo.")
        raise HTTPException(400, msg) from exc


def _dataset_payload(df: pd.DataFrame, dataset_id: str, file_name: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        "dataset_id": dataset_id,
        "file_name": file_name,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "donut": classify_columns(df),
        "preview": preview_rows(df, 5),
        "profile": to_jsonable(DataProfiler().profile(df)),
    }
    if meta:
        payload.update({"sheets": meta.get("sheets"), "sheet": meta.get("sheet")})
    return payload


def _ingest(df: pd.DataFrame, file_name: str, meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    try:
        DataIngestor().validate(df)
    except ValueError as exc:  # Spanish, user-facing
        raise HTTPException(400, str(exc)) from exc
    dataset_id = datasets.add(df, file_name)
    return _dataset_payload(df, dataset_id, file_name, meta)


@router.post("/upload")
async def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    content = await file.read()
    df, meta = _read_upload(file.filename or "archivo", content)
    return _ingest(df, file.filename or "archivo", meta)


@router.post("/sample")
def load_sample() -> Dict[str, Any]:
    if not SAMPLE_PATH.exists():
        raise HTTPException(500, "No se encontró el dataset de ejemplo.")
    with open(SAMPLE_PATH, "rb") as f:
        df, meta = _read_upload(SAMPLE_PATH.name, f.read())
    return _ingest(df, SAMPLE_PATH.name, meta)


def _require_df(dataset_id: str) -> pd.DataFrame:
    df = datasets.get_df(dataset_id)
    if df is None:
        raise HTTPException(404, "Dataset no encontrado o expirado. Vuelve a subir el archivo.")
    return df


@router.get("/{dataset_id}")
def get_dataset(dataset_id: str) -> Dict[str, Any]:
    df = _require_df(dataset_id)
    return _dataset_payload(df, dataset_id, datasets.get(dataset_id)["file_name"])


@router.post("/{dataset_id}/suggest-columns")
def suggest(dataset_id: str, body: ContextBody) -> Dict[str, Any]:
    df = _require_df(dataset_id)
    result = suggest_columns(df, dataset_context=body.context or None)
    filtered_df = result["filtered_df"]
    return to_jsonable({
        "available_columns": list(filtered_df.columns),
        "static_filter_result": result["static_filter_result"],
        "column_recommendation": result["column_recommendation"],
        "llm_error": result["llm_error"],
    })


@router.post("/{dataset_id}/chat")
def chat(dataset_id: str, body: ChatBody) -> Dict[str, Any]:
    df = _require_df(dataset_id)
    result = answer_data_question(
        df, body.question, context=body.context, mode="raw", history=body.history,
    )
    return serialize_qa_result(result)
