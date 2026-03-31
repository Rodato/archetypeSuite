from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class KOptimizer:
    """Determina el número óptimo de clusters mediante Elbow Method y Silhouette Analysis."""

    def __init__(self, k_min: int = 2, k_max: int = 10, random_state: int = 42):
        self.k_min = k_min
        self.k_max = k_max
        self.random_state = random_state

    def analyze(self, data: np.ndarray) -> Dict[str, Any]:
        n_samples = data.shape[0]
        # No tiene sentido buscar más clusters que muestras/10
        effective_k_max = min(self.k_max, n_samples // 10)
        effective_k_max = max(effective_k_max, self.k_min)

        k_range: List[int] = list(range(self.k_min, effective_k_max + 1))
        inertias: List[float] = []
        silhouette_scores: List[float] = []

        for k in k_range:
            km = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
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
