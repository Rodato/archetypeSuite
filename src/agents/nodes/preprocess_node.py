import pandas as pd

from src.data.preprocess_strategy import derive_preprocess_strategy
from src.data.preprocessor import DataPreprocessor
from src.models.state import PipelineState

# Estrategia SEGURA por defecto si la determinista fallara sobre datos públicos raros.
_SAFE_STRATEGY = {
    "drop_columns": [],
    "imputation": "median",
    "scaling": "standard",
    "encoding": "onehot",
    "dimensionality_reduction": None,
    "ordinal_mappings": {},
}


def preprocess_node(state: PipelineState) -> dict:
    """Preprocesamiento DETERMINISTA (sin LLM): misma entrada → misma estrategia → mismos clusters.

    Antes un LLM decidía scaling/encoding/drops (el último modelo dentro de la capa que
    promete determinismo). Ahora la estrategia se deriva de los datos; el único insumo
    semántico —qué texto es ordinal y en qué orden— llega curado en `ordinal_mappings`.
    """
    df = pd.DataFrame(state["raw_data"])
    original_cols = df.columns.tolist()
    ordinal_mappings = state.get("ordinal_mappings") or {}

    strategy = derive_preprocess_strategy(df, ordinal_mappings)
    preprocessor = DataPreprocessor()

    logs: list = []
    try:
        processed_df, meta = preprocessor.preprocess(df, strategy)
    except Exception as exc:  # noqa: BLE001 — red de seguridad para datos públicos arbitrarios
        processed_df, meta = preprocessor.preprocess(df, dict(_SAFE_STRATEGY))
        strategy = dict(_SAFE_STRATEGY)
        logs.append(f"[preprocess] Estrategia determinista falló ({exc}); se usó la segura por defecto.")

    ordinal_applied = meta.get("ordinal_encoded") or []
    logs.append(
        f"[preprocess] Estrategia determinista: scaling={strategy['scaling']}, encoding=onehot"
        + (f", ordinales={ordinal_applied}" if ordinal_applied else "")
    )
    logs.append(f"[preprocess] Result shape: {processed_df.shape}")

    return {
        "preprocess_strategy": strategy,
        "processed_data": processed_df.to_dict(orient="list"),
        "original_columns": original_cols,
        "log_messages": logs,
    }
