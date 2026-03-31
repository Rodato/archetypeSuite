import json

from src.clustering.registry import AlgorithmRegistry
from src.llm.prompts import REFINEMENT_PROMPT
from src.llm.provider import extract_json, get_llm_json
from src.models.schemas import RefinementDecision
from src.models.state import PipelineState


def refinement_node(state: PipelineState) -> dict:
    llm = get_llm_json()
    registry = AlgorithmRegistry()
    refinement_count = state.get("refinement_count", 0)

    context = state.get("dataset_context") or "No se proporcionó contexto adicional."
    prompt = REFINEMENT_PROMPT.format(
        metrics=json.dumps(state["metrics"], indent=2, default=str),
        n_clusters=state["n_clusters"],
        algorithm=state["selected_algorithm"],
        params=json.dumps(state.get("algorithm_params", {}), default=str),
        refinement_count=refinement_count,
        context=context,
        algorithm_descriptions=registry.get_descriptions_for_llm(),
    )

    response = llm.invoke(prompt)
    decision = RefinementDecision(**json.loads(extract_json(response.content)))

    result: dict = {
        "should_refine": decision.should_refine,
        "refinement_reason": decision.reason,
        "refinement_count": refinement_count + 1,
        "log_messages": [
            f"[refinement] Should refine: {decision.should_refine} - {decision.reason}"
        ],
    }

    valid_algorithms = set(registry.list_algorithms().keys())
    if decision.should_refine and decision.suggested_algorithm:
        if decision.suggested_algorithm in valid_algorithms:
            result["selected_algorithm"] = decision.suggested_algorithm
        else:
            result["selected_algorithm"] = "KMeans"
            result["log_messages"].append(
                f"[refinement] Algoritmo sugerido '{decision.suggested_algorithm}' "
                f"no disponible, usando KMeans"
            )
    if decision.should_refine and decision.suggested_params:
        result["algorithm_params"] = decision.suggested_params

    return result
