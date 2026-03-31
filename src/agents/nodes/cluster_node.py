import pandas as pd

from src.clustering.executor import ClusteringExecutor
from src.clustering.registry import AlgorithmRegistry
from src.models.state import PipelineState


ALLOWED_ALGORITHMS = {"KMeans", "AgglomerativeClustering"}


def cluster_node(state: PipelineState) -> dict:
    registry = AlgorithmRegistry()
    executor = ClusteringExecutor(registry)

    algorithm = state["selected_algorithm"]
    if algorithm not in ALLOWED_ALGORITHMS:
        algorithm = "KMeans"

    processed_df = pd.DataFrame(state["processed_data"])
    params = dict(state.get("algorithm_params") or {})
    optimal_k = state.get("optimal_k")
    if optimal_k:
        params["n_clusters"] = optimal_k

    result = executor.execute(
        name=algorithm,
        data=processed_df.values,
        params=params,
    )

    labels = [int(x) for x in result["labels"]]
    unique_labels = set(labels) - {-1}

    return {
        "labels": labels,
        "n_clusters": len(unique_labels),
        "log_messages": [
            f"[cluster] Executed {result['algorithm']}, "
            f"found {len(unique_labels)} clusters"
        ],
    }
