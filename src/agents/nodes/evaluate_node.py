import numpy as np
import pandas as pd

from src.clustering.evaluator import ClusteringEvaluator
from src.models.state import PipelineState


def evaluate_node(state: PipelineState) -> dict:
    processed_df = pd.DataFrame(state["processed_data"])
    labels = np.array(state["labels"])

    evaluator = ClusteringEvaluator()
    metrics = evaluator.evaluate(processed_df.values, labels)

    original_df = pd.DataFrame(state["raw_data"])
    cluster_profiles = evaluator.compute_cluster_profiles(original_df, labels)

    return {
        "metrics": metrics,
        "cluster_profiles": cluster_profiles,
        "log_messages": [
            f"[evaluate] Silhouette: {metrics.get('silhouette_score'):.3f}"
            if metrics.get("silhouette_score") is not None
            else "[evaluate] Silhouette: N/A",
            f"[evaluate] Calinski-Harabasz: {metrics.get('calinski_harabasz_score'):.1f}"
            if metrics.get("calinski_harabasz_score") is not None
            else "[evaluate] Calinski-Harabasz: N/A",
        ],
    }
