import json

import pandas as pd

from src.data.preprocessor import DataPreprocessor
from src.llm.prompts import PREPROCESSING_PROMPT
from src.llm.provider import get_llm_json, invoke_json_with_retry
from src.models.schemas import PreprocessingDecision
from src.models.state import PipelineState


def preprocess_node(state: PipelineState) -> dict:
    llm = get_llm_json()

    profile_str = json.dumps(state["profile"], indent=2, default=str)
    context = state.get("dataset_context") or "No se proporcionó contexto adicional."
    prompt = PREPROCESSING_PROMPT.format(profile=profile_str, context=context)

    decision, error = invoke_json_with_retry(
        llm, prompt, PreprocessingDecision, PreprocessingDecision
    )

    strategy = {
        "drop_columns": decision.drop_columns,
        "imputation": decision.imputation,
        "scaling": decision.scaling,
        "encoding": decision.encoding,
        "dimensionality_reduction": decision.dimensionality_reduction,
    }

    df = pd.DataFrame(state["raw_data"])
    original_cols = df.columns.tolist()
    preprocessor = DataPreprocessor()

    logs: list = []
    try:
        processed_df, _ = preprocessor.preprocess(df, strategy)
        logs.append(
            f"[preprocess] LLM strategy: scaling={decision.scaling}, "
            f"encoding={decision.encoding}, dropped={decision.drop_columns}"
        )
    except Exception as exc:  # noqa: BLE001 — fall back to a known-safe strategy
        safe_strategy = {
            "drop_columns": [],
            "imputation": "median",
            "scaling": "standard",
            "encoding": "onehot",
            "dimensionality_reduction": None,
        }
        processed_df, _ = preprocessor.preprocess(df, safe_strategy)
        strategy = safe_strategy
        logs.append(f"[preprocess] Estrategia del LLM falló ({exc}); se usó una estrategia segura por defecto.")

    logs.append(f"[preprocess] Result shape: {processed_df.shape}")
    if error:
        logs.insert(0, f"[preprocess] LLM falló, usando defaults. Error: {error}")

    return {
        "preprocess_strategy": strategy,
        "processed_data": processed_df.to_dict(orient="list"),
        "original_columns": original_cols,
        "log_messages": logs,
    }
