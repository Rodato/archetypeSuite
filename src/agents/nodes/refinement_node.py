"""Gate determinista de refinamiento.

Antes era un LLM decidiendo si re-clusterizar — el único punto del pipeline donde
un modelo tomaba una decisión numérica. Ahora es una regla por umbral de silhouette
con un único reintento más exhaustivo (n_init alto): determinista, gratis y auditable.
"""
from src.config.settings import settings
from src.models.state import PipelineState


def refinement_node(state: PipelineState) -> dict:
    refinement_count = state.get("refinement_count", 0)
    metrics = state.get("metrics") or {}
    silhouette = metrics.get("silhouette_score")
    threshold = settings.refinement_silhouette_threshold

    refine = (
        refinement_count == 0
        and isinstance(silhouette, (int, float))
        and silhouette < threshold
    )

    if refine:
        reason = (
            f"Separación débil (silhouette {silhouette:.2f} < {threshold}): se reintenta "
            f"el clustering con búsqueda más exhaustiva (n_init={settings.refinement_n_init})."
        )
        return {
            "should_refine": True,
            "refinement_reason": reason,
            "refinement_count": refinement_count + 1,
            "algorithm_params": {
                **state.get("algorithm_params", {}),
                "n_init": settings.refinement_n_init,
            },
            "log_messages": [f"[refinement] {reason}"],
        }

    if refinement_count > 0:
        reason = "Ya se aplicó el reintento exhaustivo; se conserva este resultado."
    elif isinstance(silhouette, (int, float)):
        reason = (
            f"Separación aceptable (silhouette {silhouette:.2f} ≥ {threshold}): "
            "no hace falta refinar."
        )
    else:
        reason = "Sin métrica de separación disponible; no se refina."

    return {
        "should_refine": False,
        "refinement_reason": reason,
        "refinement_count": refinement_count + 1,
        "log_messages": [f"[refinement] {reason}"],
    }
