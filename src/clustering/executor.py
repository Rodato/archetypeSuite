from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.clustering.registry import AlgorithmRegistry
from src.config.settings import settings


class ClusteringExecutor:
    def __init__(self, registry: AlgorithmRegistry) -> None:
        self._registry = registry

    def execute(
        self,
        name: str,
        data: np.ndarray | pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        algorithm_entry = self._registry.get(name)

        merged_params = {**algorithm_entry["default_params"]}
        if params:
            merged_params.update(params)

        # El seed es invariante del producto (determinismo): ningún caller puede
        # sobreescribirlo, venga de donde venga el dict de params.
        if "random_state" in merged_params:
            merged_params["random_state"] = settings.random_seed

        model = algorithm_entry["class"](**merged_params)
        labels = model.fit_predict(data)

        return {
            "labels": labels.tolist(),
            "model": model,
            "algorithm": name,
            "params": merged_params,
        }
