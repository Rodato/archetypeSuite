import pandas as pd

from src.config.settings import settings
from src.data.k_optimizer import KOptimizer
from src.models.state import PipelineState


def optimize_k_node(state: PipelineState) -> dict:
    processed_df = pd.DataFrame(state["processed_data"])
    n_samples = processed_df.shape[0]
    n_features = processed_df.shape[1]

    if n_samples < 4:
        raise ValueError(
            f"El dataset es demasiado pequeño para clustering ({n_samples} filas tras "
            "preprocesamiento). Necesitamos al menos 4 filas con valores válidos."
        )
    if n_features < 1:
        raise ValueError(
            "No quedaron columnas útiles tras el preprocesamiento. "
            "Revisa la selección de variables y el filtrado estático."
        )

    k_max = min(settings.k_optimizer_max, max(settings.k_optimizer_min, n_samples // 10))
    optimizer = KOptimizer(k_min=settings.k_optimizer_min, k_max=k_max)
    analysis = optimizer.analyze(processed_df.values)

    return {
        "optimal_k": analysis["optimal_k"],
        "k_analysis": analysis,
        "log_messages": [
            f"[optimize_k] Silhouette Analysis → k óptimo = {analysis['best_silhouette_k']} "
            f"(score={analysis['best_silhouette_score']:.3f})",
            f"[optimize_k] Elbow Method → k sugerido = {analysis['elbow_k']}",
            f"[optimize_k] k final seleccionado = {analysis['optimal_k']} (por Silhouette)",
        ],
    }
