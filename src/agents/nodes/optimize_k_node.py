import pandas as pd

from src.data.k_optimizer import KOptimizer
from src.models.state import PipelineState


def optimize_k_node(state: PipelineState) -> dict:
    processed_df = pd.DataFrame(state["processed_data"])
    n_samples = processed_df.shape[0]

    k_max = min(10, n_samples // 10)
    optimizer = KOptimizer(k_min=2, k_max=k_max)
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
