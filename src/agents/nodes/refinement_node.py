import json

from src.llm.prompts import REFINEMENT_PROMPT
from src.llm.provider import get_llm_json, invoke_json_with_retry
from src.models.schemas import RefinementDecision
from src.models.state import PipelineState

# Solo hiperparámetros seguros de KMeans pueden venir del LLM. Sin whitelist, una
# sugerencia como {"algorithm": "DBSCAN"} (kwarg válido de KMeans, valor inválido) o
# {"random_state": null} crashea la segunda pasada de cluster o rompe el determinismo.
_ALLOWED_PARAMS = {"init", "n_init", "max_iter"}
_VALID_INIT = {"k-means++", "random"}


def _sanitize_suggested_params(suggested: dict) -> dict:
    safe: dict = {}
    for key, value in suggested.items():
        if key not in _ALLOWED_PARAMS:
            continue
        if key == "init":
            if value in _VALID_INIT:
                safe[key] = value
        elif isinstance(value, int) and not isinstance(value, bool) and value > 0:
            safe[key] = value
    return safe


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
        params = _sanitize_suggested_params(decision.suggested_params)
        dropped = set(decision.suggested_params) - set(params)
        if dropped:
            logs.append(f"[refinement] Parámetros descartados por whitelist: {sorted(dropped)}")
        if params:
            result["algorithm_params"] = {**state.get("algorithm_params", {}), **params}

    return result
