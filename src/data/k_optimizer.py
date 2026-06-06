from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config.settings import settings


class KOptimizer:
    """Determina el número óptimo de clusters mediante Elbow Method y Silhouette Analysis."""

    def __init__(self, k_min: int = None, k_max: int = None, random_state: int = None):
        self.k_min = k_min if k_min is not None else settings.k_optimizer_min
        self.k_max = k_max if k_max is not None else settings.k_optimizer_max
        self.random_state = random_state if random_state is not None else settings.random_seed

    def analyze(self, data: np.ndarray) -> Dict[str, Any]:
        n_samples = data.shape[0]
        # Aim for at least ~5 samples per cluster, but never below k_min and never k >= n.
        effective_k_max = min(self.k_max, max(self.k_min, n_samples // 5), n_samples - 1)
        effective_k_max = max(effective_k_max, self.k_min)

        k_range: List[int] = list(range(self.k_min, effective_k_max + 1))
        forced_k_min = len(k_range) <= 1
        inertias: List[float] = []
        silhouette_scores: List[float] = []

        for k in k_range:
            km = KMeans(n_clusters=k, n_init=settings.kmeans_n_init, random_state=self.random_state)
            labels = km.fit_predict(data)
            inertias.append(float(km.inertia_))
            silhouette_scores.append(float(silhouette_score(data, labels)))

        best_sil_idx = int(np.argmax(silhouette_scores))
        best_silhouette_k = k_range[best_sil_idx]
        best_silhouette_score = silhouette_scores[best_sil_idx]

        elbow_k = self._find_elbow(k_range, inertias)

        return {
            "k_range": k_range,
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "best_silhouette_k": best_silhouette_k,
            "best_silhouette_score": best_silhouette_score,
            "elbow_k": elbow_k,
            "optimal_k": best_silhouette_k,
            "forced_k_min": forced_k_min,
        }

    def _find_elbow(self, k_range: List[int], inertias: List[float]) -> int:
        """Detecta el codo geométricamente: punto de máxima distancia a la línea recta."""
        if len(k_range) < 3:
            return k_range[0]

        k_arr = np.array(k_range, dtype=float)
        i_arr = np.array(inertias, dtype=float)

        # Normalizar ambos ejes a [0, 1]
        k_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min() or 1.0)
        i_norm = (i_arr - i_arr.min()) / (i_arr.max() - i_arr.min() or 1.0)

        # Vector de la línea recta entre el primer y último punto
        line = np.array([k_norm[-1] - k_norm[0], i_norm[-1] - i_norm[0]])
        line_norm = line / (np.linalg.norm(line) or 1.0)

        distances = []
        for idx in range(len(k_range)):
            point = np.array([k_norm[idx] - k_norm[0], i_norm[idx] - i_norm[0]])
            proj = np.dot(point, line_norm) * line_norm
            distances.append(float(np.linalg.norm(point - proj)))

        return k_range[int(np.argmax(distances))]
