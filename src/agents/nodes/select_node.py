import json

import pandas as pd

from src.clustering.registry import AlgorithmRegistry
from src.llm.prompts import ALGORITHM_SELECTION_PROMPT
from src.llm.provider import extract_json, get_llm_json
from src.models.schemas import AlgorithmSelection
from src.models.state import PipelineState


def select_node(state: PipelineState) -> dict:
    llm = get_llm_json()
    registry = AlgorithmRegistry()

    processed_df = pd.DataFrame(state["processed_data"])
    metadata = state.get("preprocessing_metadata", {})

    context = state.get("dataset_context") or "No se proporcionó contexto adicional."
    k_analysis = state.get("k_analysis", {})
    prompt = ALGORITHM_SELECTION_PROMPT.format(
        n_rows=processed_df.shape[0],
        n_cols=processed_df.shape[1],
        n_numeric=processed_df.shape[1],
        preprocessing_metadata=json.dumps(metadata, default=str),
        algorithm_descriptions=registry.get_descriptions_for_llm(),
        context=context,
        optimal_k=state.get("optimal_k", 4),
        best_silhouette_k=k_analysis.get("best_silhouette_k", "N/A"),
        best_silhouette_score=k_analysis.get("best_silhouette_score", 0.0),
        elbow_k=k_analysis.get("elbow_k", "N/A"),
    )

    response = llm.invoke(prompt)
    selection = AlgorithmSelection(**json.loads(extract_json(response.content)))

    return {
        "selected_algorithm": selection.algorithm,
        "algorithm_params": selection.params,
        "selection_reasoning": selection.reasoning,
        "log_messages": [
            f"[select] LLM selected: {selection.algorithm} "
            f"with params {selection.params}",
            f"[select] Reasoning: {selection.reasoning}",
        ],
    }
