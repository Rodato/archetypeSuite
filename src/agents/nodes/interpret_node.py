import json

from src.llm.prompts import INTERPRETATION_PROMPT
from src.llm.provider import get_narrative_llm, invoke_json_with_retry
from src.models.schemas import ArchetypeDescription, InterpretationResult
from src.models.state import PipelineState


def _fallback_interpretation(n_clusters: int) -> InterpretationResult:
    archetypes = [
        ArchetypeDescription(
            cluster_id=i,
            label=f"Cluster {i}",
            description="No se pudo generar descripción narrativa (fallo del LLM).",
            key_characteristics=[],
            differentiators=[],
        )
        for i in range(n_clusters)
    ]
    return InterpretationResult(
        archetypes=archetypes,
        summary="No se pudo generar resumen (fallo del LLM).",
    )


def interpret_node(state: PipelineState) -> dict:
    llm = get_narrative_llm()
    n_clusters = state["n_clusters"]

    context = state.get("dataset_context") or "No se proporcionó contexto adicional."
    prompt = INTERPRETATION_PROMPT.format(
        cluster_profiles=json.dumps(state["cluster_profiles"], indent=2, default=str),
        metrics=json.dumps(state["metrics"], indent=2, default=str),
        n_clusters=n_clusters,
        original_columns=json.dumps(state.get("original_columns", [])),
        context=context,
    )

    result, error = invoke_json_with_retry(
        llm,
        prompt,
        InterpretationResult,
        lambda: _fallback_interpretation(n_clusters),
    )

    archetypes = [a.model_dump() for a in result.archetypes]

    logs = [
        f"[interpret] Generated {len(archetypes)} archetype descriptions",
        f"[interpret] Summary: {result.summary[:100]}..."
        if len(result.summary) > 100
        else f"[interpret] Summary: {result.summary}",
    ]
    if error:
        logs.insert(0, f"[interpret] LLM falló, usando placeholders. Error: {error}")

    return {
        "archetypes": archetypes,
        "log_messages": logs,
    }
