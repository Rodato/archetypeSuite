import json

from src.llm.prompts import REFINEMENT_PROMPT
from src.llm.provider import get_llm_json, invoke_json_with_retry
from src.models.schemas import RefinementDecision
from src.models.state import PipelineState


def refinement_node(state: PipelineState) -> dict:
    llm = get_llm_json()
    refinement_count = state.get("refinement_count", 0)

    context = state.get("dataset_context") or "No se proporcionó contexto adicional."
    prompt = REFINEMENT_PROMPT.format(
        metrics=json.dumps(state["metrics"], indent=2, default=str),
        n_clusters=state["n_clusters"],
        algorithm=state["selected_algorithm"],
        params=json.dumps(state.get("algorithm_params", {}), default=str),
        refinement_count=refinement_count,
        context=context,
    )

    decision, error = invoke_json_with_retry(
        llm,
        prompt,
        RefinementDecision,
        lambda: RefinementDecision(should_refine=False, reason="LLM falló, no se refinará"),
    )

    logs = [
        f"[refinement] Should refine: {decision.should_refine} - {decision.reason}"
    ]
    if error:
        logs.insert(0, f"[refinement] LLM falló, usando fallback. Error: {error}")

    result: dict = {
        "should_refine": decision.should_refine,
        "refinement_reason": decision.reason,
        "refinement_count": refinement_count + 1,
        "log_messages": logs,
    }

    if decision.should_refine and decision.suggested_params:
        params = dict(decision.suggested_params)
        params.pop("n_clusters", None)
        if params:
            result["algorithm_params"] = {**state.get("algorithm_params", {}), **params}

    return result
