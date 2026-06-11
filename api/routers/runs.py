"""Run endpoints: list / get / archetypes-mode chat / exports / curation / delete."""
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

import json

from fastapi.responses import StreamingResponse

from api.store import delete_run, get_run, list_runs, new_id, now_iso, save_run
from api.transform import label_map_from_archetypes, serialize_qa_result
from src.llm.chat_agent import answer_chat, stream_chat
from src.llm.group_profile import profile_group
from src.core.export import archetypes_to_csv, build_markdown_report, labels_to_csv

router = APIRouter(prefix="/api/runs", tags=["runs"])


class ChatBody(BaseModel):
    question: str
    context: str = ""
    history: Optional[List[Dict[str, str]]] = None


class ArchetypeEdit(BaseModel):
    """Campos curables por el equipo. `nivel_cautela` NO es editable: su piso lo fija
    determinísticamente la calidad del clustering (caution_from_silhouette)."""
    label: Optional[str] = Field(None, min_length=1, max_length=120)
    description: Optional[str] = Field(None, max_length=2000)
    comportamiento_principal: Optional[str] = Field(None, max_length=1000)
    microcomportamientos: Optional[List[str]] = None
    barreras: Optional[List[str]] = None
    habilitadores: Optional[List[str]] = None
    oportunidades_accion: Optional[List[str]] = None
    cautela_reason: Optional[str] = Field(None, max_length=1000)
    validated: Optional[bool] = None


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


class ProfileGroupBody(BaseModel):
    description: str = Field(min_length=3, max_length=500)


@router.post("/{run_id}/profile-group")
def create_group_profile(run_id: str, body: ProfileGroupBody) -> Dict[str, Any]:
    """Perfilado a demanda: hipótesis comportamental para un grupo descrito en lenguaje
    natural — funciona aunque el grupo no coincida con ningún arquetipo del clustering."""
    record = _require_run(run_id)
    df = _reconstruct_labeled_df(record)
    if df.empty:
        raise HTTPException(422, "Este análisis no tiene datos para perfilar.")
    result = profile_group(df, body.description, context=record.get("dataset_context", ""))
    if result.error:
        raise HTTPException(422, result.error)
    profile = {
        "id": new_id(),
        "created_at": now_iso(),
        "origin": "user_defined",
        "group_description": body.description,
        "interpretation": result.interpretation,
        "filters": result.filters,
        "n": result.n,
        "share": result.share,
        **(result.profile or {}),
    }
    record.setdefault("custom_profiles", []).append(profile)
    save_run(record)
    return profile


@router.delete("/{run_id}/profiles/{profile_id}")
def delete_group_profile(run_id: str, profile_id: str) -> Dict[str, Any]:
    record = _require_run(run_id)
    profiles = record.get("custom_profiles", []) or []
    kept = [p for p in profiles if p.get("id") != profile_id]
    if len(kept) == len(profiles):
        raise HTTPException(404, "Perfil no encontrado.")
    record["custom_profiles"] = kept
    save_run(record)
    return {"ok": True}


_EDITABLE_LIST_FIELDS = ("microcomportamientos", "barreras", "habilitadores", "oportunidades_accion")


@router.patch("/{run_id}/archetypes/{cluster_id}")
def edit_archetype(run_id: str, cluster_id: int, body: ArchetypeEdit) -> Dict[str, Any]:
    """Curación humana: el equipo edita la hipótesis que propuso el LLM y la marca validada."""
    record = _require_run(run_id)
    target = next(
        (a for a in record.get("archetypes", []) if a.get("cluster_id") == cluster_id), None,
    )
    if target is None:
        raise HTTPException(404, "Arquetipo no encontrado en este análisis.")

    changes = body.model_dump(exclude_unset=True)
    validated = changes.pop("validated", None)
    for field, value in changes.items():
        if field in _EDITABLE_LIST_FIELDS:
            value = [str(v).strip() for v in value if str(v).strip()]
        target[field] = value
    if validated is not None:
        target["validated"] = validated
    if changes or validated is not None:
        target["curated_at"] = now_iso()

    # El label vive denormalizado en sizes y charts — propagarlo para que toda la UI
    # (barras, radar, mapa, box) hable con el nombre curado.
    new_label = changes.get("label")
    if new_label:
        for row in record.get("cluster_sizes", []) or []:
            if row.get("cluster_id") == cluster_id:
                row["label"] = new_label
        charts = record.get("charts", {}) or {}
        for serie in (charts.get("radar", {}) or {}).get("series", []) or []:
            if serie.get("cluster_id") == cluster_id:
                serie["label"] = new_label
        for point in charts.get("scatter", []) or []:
            if point.get("cluster_id") == cluster_id:
                point["archetype"] = new_label
        for groups in (charts.get("box", {}) or {}).values():
            for g in groups or []:
                if g.get("cluster_id") == cluster_id:
                    g["label"] = new_label

    save_run(record)
    return target


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
    result = answer_chat(
        df, body.question,
        context=body.context or record.get("dataset_context", ""),
        mode="archetypes", history=body.history,
        archetypes=record.get("archetypes"),
    )
    return serialize_qa_result(result)


def sse_chat_stream(events) -> StreamingResponse:
    """Retransmite los eventos del agente como SSE (tools en vivo + resultado final)."""
    def gen():
        for ev in events:
            if ev["type"] == "tool":
                payload = {
                    "type": "tool",
                    "tool": ev["tool"],
                    "ok": ev["ok"],
                    "summary": (ev.get("summary") or "")[:160],
                }
            else:
                payload = {"type": "result", "payload": serialize_qa_result(ev["result"])}
            yield f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@router.post("/{run_id}/chat/stream")
def chat_stream(run_id: str, body: ChatBody) -> StreamingResponse:
    record = _require_run(run_id)
    df = _reconstruct_labeled_df(record)
    return sse_chat_stream(stream_chat(
        df, body.question,
        context=body.context or record.get("dataset_context", ""),
        mode="archetypes", history=body.history,
        archetypes=record.get("archetypes"),
    ))


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
