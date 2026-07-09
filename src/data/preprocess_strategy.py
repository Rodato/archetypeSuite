"""Estrategia de preprocesamiento DETERMINISTA (sin LLM).

Reemplaza la decisión que antes tomaba un LLM. Para clustering de encuestas la estrategia
buena es casi siempre la misma (median / one-hot / standard), así preprocess vuelve a la
capa determinista: misma entrada → misma estrategia → mismos clusters.

Sobre el escalado: se usa SIEMPRE `standard`. Se probó un auto-switch a `robust` ante
outliers, pero en datos comportamentales las colas pesadas (conteos tipo posts/interacciones,
unos pocos power-users) SON señal, no ruido — robustificarlas colapsaba el demo a un
mega-cluster. `standard` reproduce el baseline y respeta el dominio. (Si en el futuro un
dataset raro necesitara robust, será una elección curada del usuario, no un auto-switch frágil.)

El único insumo semántico —¿qué texto es ordinal y en qué orden?— NO se adivina acá: llega
como `ordinal_mappings` curado desde el wizard (paso 2).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def derive_preprocess_strategy(
    df: pd.DataFrame,
    ordinal_mappings: Optional[Dict[str, List[Any]]] = None,
) -> Dict[str, Any]:
    """Estrategia determinista (sin LLM).

    - drop_columns: [] — los filtros estáticos (`column_filter`) ya descartaron lo inservible.
    - imputation: median (numéricas) / most_frequent (categóricas, dentro del preprocesador).
    - scaling: standard (ver módulo — no se auto-selecciona robust).
    - encoding: one-hot para nominales.
    - ordinal_mappings: mapa curado {col: [orden]} → esas columnas se codifican por rango y se escalan.
    """
    # Defensivo: un `ordinal_mappings` mal formado (no-dict) → vacío, sin abortar el nodo.
    mappings = ordinal_mappings if isinstance(ordinal_mappings, dict) else {}
    # Descartar mapeos de columnas ausentes (p.ej. filtradas por column_selection o nombre
    # erróneo): evita arrastrar keys muertas en la estrategia persistida / mostrada en UI.
    mappings = {k: v for k, v in mappings.items() if k in df.columns}
    return {
        "drop_columns": [],
        "imputation": "median",
        "scaling": "standard",
        "encoding": "onehot",
        "dimensionality_reduction": None,
        "ordinal_mappings": dict(mappings),
    }
