from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config.settings import settings


def select_optimal_k(
    k_range: List[int],
    silhouette_scores: List[float],
    *,
    flat_range: float = None,
    flat_max_k: int = None,
) -> tuple:
    """Regla de dos regímenes para elegir k. Devuelve (k, flat_curve).

    - Curva con señal (max−min ≥ flat_range): argmax clásico — hay un k que separa mejor.
    - Curva plana: los datos no distinguen ningún k (el silhouette se arrastra hacia
      arriba con k por pura fragmentación). Elegir el mejor entre los "pocos y
      trabajables" (k ≤ flat_max_k) en vez de perseguir el tope del rango.
    """
    flat_range = flat_range if flat_range is not None else settings.k_flat_curve_range
    flat_max_k = flat_max_k if flat_max_k is not None else settings.k_flat_max_k

    spread = max(silhouette_scores) - min(silhouette_scores)
    flat = spread < flat_range and len(k_range) > 1
    if not flat:
        return k_range[int(np.argmax(silhouette_scores))], False

    candidates = [(k, s) for k, s in zip(k_range, silhouette_scores) if k <= flat_max_k]
    if not candidates:
        candidates = list(zip(k_range, silhouette_scores))
    best_k = max(candidates, key=lambda t: t[1])[0]
    return best_k, True


class KOptimizer:
    """Determina el número óptimo de clusters mediante Elbow Method y Silhouette Analysis."""

    def __init__(self, k_min: int = None, k_max: int = None, random_state: int = None):
        self.k_min = k_min if k_min is not None else settings.k_optimizer_min
        self.k_max = k_max if k_max is not None else settings.k_optimizer_max
        self.random_state = random_state if random_state is not None else settings.random_seed

    def analyze(self, data: np.ndarray) -> Dict[str, Any]:
        n_samples = data.shape[0]
        # Cap conservador (~10 muestras por cluster); antes vivía duplicado y en
        # conflicto entre este módulo (n//5) y optimize_k_node (n//10) — gana n//10.
        effective_k_max = min(self.k_max, max(self.k_min, n_samples // 10), n_samples - 1)
        effective_k_max = max(effective_k_max, self.k_min)

        candidate_k: List[int] = list(range(self.k_min, effective_k_max + 1))
        k_range: List[int] = []
        inertias: List[float] = []
        silhouette_scores: List[float] = []

        for k in candidate_k:
            km = KMeans(n_clusters=k, n_init=settings.kmeans_n_init, random_state=self.random_state)
            labels = km.fit_predict(data)
            # KMeans puede colapsar a <2 clusters reales (p.ej. muchas filas idénticas en
            # encuestas Likert). silhouette_score levantaría ValueError; en vez de abortar
            # TODA la búsqueda de k, se saltea ese k y se sigue con los válidos.
            if len(set(labels)) < 2:
                continue
            try:
                sil = float(silhouette_score(data, labels))
            except ValueError:
                continue
            k_range.append(k)
            inertias.append(float(km.inertia_))
            silhouette_scores.append(sil)

        if not k_range:
            raise ValueError(
                "Ningún número de grupos produjo una partición válida: los datos podrían "
                "tener demasiadas filas idénticas o muy poca variación tras el preprocesamiento."
            )
        forced_k_min = len(k_range) <= 1

        best_sil_idx = int(np.argmax(silhouette_scores))
        best_silhouette_k = k_range[best_sil_idx]
        best_silhouette_score = silhouette_scores[best_sil_idx]

        elbow_k = self._find_elbow(k_range, inertias)
        optimal_k, flat_k_curve = select_optimal_k(k_range, silhouette_scores)

        return {
            "k_range": k_range,
            "inertias": inertias,
            "silhouette_scores": silhouette_scores,
            "best_silhouette_k": best_silhouette_k,
            "best_silhouette_score": best_silhouette_score,
            "elbow_k": elbow_k,
            "optimal_k": optimal_k,
            "flat_k_curve": flat_k_curve,
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
