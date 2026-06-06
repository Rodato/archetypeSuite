import json

from src.llm.methodology import load_methodology
from src.llm.prompts import INTERPRETATION_PROMPT
from src.llm.provider import get_narrative_llm, invoke_json_with_retry
from src.models.schemas import ArchetypeDescription, InterpretationResult
from src.models.state import PipelineState
from src.ui.quality import CAUTION_ORDER, caution_from_silhouette


def _fallback_interpretation(n_clusters: int) -> InterpretationResult:
    archetypes = [
        ArchetypeDescription(
            cluster_id=i,
            label=f"Patrón {i + 1}",
            description=(
                "No se pudo generar la lectura narrativa (falló el modelo). "
                "Los datos del cluster están disponibles, pero la interpretación queda pendiente."
            ),
            nivel_cautela="alta",
            cautela_reason="La narrativa automática no se pudo generar; trátalo como lectura provisional.",
        )
        for i in range(n_clusters)
    ]
    return InterpretationResult(
        archetypes=archetypes,
        summary="No se pudo generar la lectura general (falló el modelo).",
    )


def interpret_node(state: PipelineState) -> dict:
    llm = get_narrative_llm()
    n_clusters = state["n_clusters"]
    metrics = state.get("metrics") or {}
    silhouette = metrics.get("silhouette_score")

    context = state.get("dataset_context") or "No se proporcionó contexto adicional."
    prompt = INTERPRETATION_PROMPT.format(
        methodology=load_methodology(),
        cluster_profiles=json.dumps(state["cluster_profiles"], indent=2, default=str),
        metrics=json.dumps(metrics, indent=2, default=str),
        silhouette=f"{silhouette:.3f}" if isinstance(silhouette, (int, float)) else "no disponible",
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

    # Deterministic caution floor (marco metodológico §9): silhouette débil ⇒ cautela alta.
    # Solo SUBE la cautela; nunca baja una 'alta' del modelo.
    floor = caution_from_silhouette(silhouette)
    for a in result.archetypes:
        if CAUTION_ORDER.get(a.nivel_cautela, 1) < CAUTION_ORDER[floor]:
            a.nivel_cautela = floor
            if silhouette is not None:
                note = (
                    f"La separación entre grupos (silhouette {silhouette:.2f}) es "
                    f"{'débil' if floor == 'alta' else 'moderada'}, así que la lectura se "
                    f"reporta con cautela {floor}."
                )
            else:
                note = "No se pudo medir la separación entre grupos; lectura con cautela alta."
            a.cautela_reason = f"{a.cautela_reason} {note}".strip() if a.cautela_reason else note

    archetypes = [a.model_dump() for a in result.archetypes]

    logs = [
        f"[interpret] Generated {len(archetypes)} archetype descriptions (cautela base: {floor})",
        f"[interpret] Summary: {result.summary[:100]}..."
        if len(result.summary) > 100
        else f"[interpret] Summary: {result.summary}",
    ]
    if error:
        logs.insert(0, f"[interpret] LLM falló, usando placeholders. Error: {error}")

    return {
        "archetypes": archetypes,
        "interpretation_summary": result.summary,
        "log_messages": logs,
    }
