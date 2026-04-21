from src.models.state import PipelineState


FIXED_ALGORITHM = "KMeans"


def select_node(state: PipelineState) -> dict:
    optimal_k = state.get("optimal_k", 4)
    params = {"n_clusters": optimal_k, "n_init": 10, "random_state": 42}

    return {
        "selected_algorithm": FIXED_ALGORITHM,
        "algorithm_params": params,
        "selection_reasoning": (
            f"Algoritmo fijado a {FIXED_ALGORITHM} para garantizar consistencia entre "
            f"corridas. k={optimal_k} determinado por Silhouette Analysis."
        ),
        "log_messages": [
            f"[select] Algoritmo fijo: {FIXED_ALGORITHM} con n_clusters={optimal_k}",
        ],
    }
