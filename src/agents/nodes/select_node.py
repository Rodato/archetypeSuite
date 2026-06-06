from src.config.settings import settings
from src.models.state import PipelineState


def select_node(state: PipelineState) -> dict:
    optimal_k = state.get("optimal_k", 4)
    algorithm = settings.fixed_algorithm
    # Only KMeans accepts n_init/random_state; build params per algorithm so a future
    # AgglomerativeClustering path wouldn't crash on unexpected kwargs.
    params = {"n_clusters": optimal_k}
    if algorithm == "KMeans":
        params["n_init"] = settings.kmeans_n_init
        params["random_state"] = settings.random_seed

    return {
        "selected_algorithm": algorithm,
        "algorithm_params": params,
        "selection_reasoning": (
            f"Algoritmo fijado a {algorithm} para garantizar consistencia entre "
            f"corridas. k={optimal_k} determinado por Silhouette Analysis."
        ),
        "log_messages": [
            f"[select] Algoritmo fijo: {algorithm} con n_clusters={optimal_k}",
        ],
    }
