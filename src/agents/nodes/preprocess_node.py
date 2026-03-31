import json

import pandas as pd

from src.data.preprocessor import DataPreprocessor
from src.llm.prompts import PREPROCESSING_PROMPT
from src.llm.provider import extract_json, get_llm_json
from src.models.schemas import PreprocessingDecision
from src.models.state import PipelineState


def preprocess_node(state: PipelineState) -> dict:
    llm = get_llm_json()

    profile_str = json.dumps(state["profile"], indent=2, default=str)
    context = state.get("dataset_context") or "No se proporcionó contexto adicional."
    prompt = PREPROCESSING_PROMPT.format(profile=profile_str, context=context)

    response = llm.invoke(prompt)
    decision = PreprocessingDecision(**json.loads(extract_json(response.content)))

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
    processed_df, metadata = preprocessor.preprocess(df, strategy)

    return {
        "preprocess_strategy": strategy,
        "processed_data": processed_df.to_dict(orient="list"),
        "preprocessing_metadata": metadata,
        "original_columns": original_cols,
        "log_messages": [
            f"[preprocess] LLM strategy: scaling={decision.scaling}, "
            f"encoding={decision.encoding}, dropped={decision.drop_columns}",
            f"[preprocess] Result shape: {processed_df.shape}",
        ],
    }
