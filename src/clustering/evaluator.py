from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

logger = logging.getLogger(__name__)


class ClusteringEvaluator:
    def evaluate(
        self,
        data: np.ndarray | pd.DataFrame,
        labels: list[int] | np.ndarray,
    ) -> dict[str, Any]:
        labels_array = np.asarray(labels)
        unique_labels = set(labels_array)
        unique_labels_no_noise = unique_labels - {-1}
        n_clusters = len(unique_labels_no_noise)
        noise_points = int(np.sum(labels_array == -1))
        cluster_sizes = {
            int(label): int(np.sum(labels_array == label))
            for label in sorted(unique_labels)
        }

        if n_clusters < 2:
            logger.warning(
                "Fewer than 2 clusters found (excluding noise). "
                "Clustering metrics cannot be computed."
            )
            return {
                "silhouette_score": None,
                "calinski_harabasz_score": None,
                "davies_bouldin_score": None,
                "n_clusters": n_clusters,
                "cluster_sizes": cluster_sizes,
                "noise_points": noise_points,
                "warning": (
                    "Fewer than 2 clusters found (excluding noise). "
                    "Metrics could not be computed."
                ),
            }

        mask = labels_array != -1
        filtered_data = np.asarray(data)[mask] if noise_points > 0 else np.asarray(data)
        filtered_labels = labels_array[mask] if noise_points > 0 else labels_array

        sil_score: float | None = None
        try:
            sil_score = float(silhouette_score(filtered_data, filtered_labels))
        except Exception as e:
            logger.warning("Failed to compute silhouette_score: %s", e)

        ch_score: float | None = None
        try:
            ch_score = float(calinski_harabasz_score(filtered_data, filtered_labels))
        except Exception as e:
            logger.warning("Failed to compute calinski_harabasz_score: %s", e)

        db_score: float | None = None
        try:
            db_score = float(davies_bouldin_score(filtered_data, filtered_labels))
        except Exception as e:
            logger.warning("Failed to compute davies_bouldin_score: %s", e)

        return {
            "silhouette_score": sil_score,
            "calinski_harabasz_score": ch_score,
            "davies_bouldin_score": db_score,
            "n_clusters": n_clusters,
            "cluster_sizes": cluster_sizes,
            "noise_points": noise_points,
        }

    def compute_cluster_profiles(
        self,
        original_df: pd.DataFrame,
        labels: list[int] | np.ndarray,
    ) -> dict[int, dict[str, dict[str, Any]]]:
        labels_array = np.asarray(labels)
        df = original_df.copy()
        df["_cluster_label"] = labels_array

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "_cluster_label"]
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        profiles: dict[int, dict[str, dict[str, Any]]] = {}

        for label in sorted(df["_cluster_label"].unique()):
            cluster_data = df[df["_cluster_label"] == label]
            label_key = int(label)
            profiles[label_key] = {}

            for col in numeric_cols:
                col_data = cluster_data[col]
                profiles[label_key][col] = {
                    "mean": float(col_data.mean()) if not col_data.isna().all() else None,
                    "median": float(col_data.median()) if not col_data.isna().all() else None,
                    "std": float(col_data.std()) if not col_data.isna().all() else None,
                }

            for col in categorical_cols:
                col_data = cluster_data[col]
                value_counts = col_data.value_counts()
                mode_value = value_counts.index[0] if len(value_counts) > 0 else None
                distribution = {
                    str(k): int(v) for k, v in value_counts.items()
                }
                profiles[label_key][col] = {
                    "mode": mode_value,
                    "distribution": distribution,
                }

        return profiles
