from __future__ import annotations

from typing import Any

from sklearn.cluster import AgglomerativeClustering, KMeans


class AlgorithmRegistry:
    _algorithms: dict[str, dict[str, Any]] = {
        "KMeans": {
            "class": KMeans,
            "description": (
                "Centroid-based clustering that partitions data into k clusters "
                "by minimizing within-cluster variance. Works best with spherical, "
                "evenly-sized clusters and requires specifying the number of clusters upfront."
            ),
            "default_params": {
                "n_clusters": 4,
                "n_init": 10,
                "random_state": 42,
            },
        },
        "AgglomerativeClustering": {
            "class": AgglomerativeClustering,
            "description": (
                "Hierarchical bottom-up clustering that starts with each point as its own "
                "cluster and iteratively merges the closest pairs. Produces a dendrogram "
                "and works well when cluster hierarchy matters. Linkage method controls "
                "how distances between clusters are computed."
            ),
            "default_params": {
                "n_clusters": 4,
                "linkage": "ward",
            },
        },
    }

    def get(self, name: str) -> dict[str, Any]:
        if name not in self._algorithms:
            raise KeyError(
                f"Algorithm '{name}' not found. "
                f"Available algorithms: {', '.join(self._algorithms.keys())}"
            )
        return self._algorithms[name]
