import json

from src.llm.prompts import INTERPRETATION_PROMPT
from src.llm.provider import extract_json, get_narrative_llm
from src.models.schemas import InterpretationResult
from src.models.state import PipelineState


def interpret_node(state: PipelineState) -> dict:
    llm = get_narrative_llm()

    context = state.get("dataset_context") or "No se proporcionó contexto adicional."
    prompt = INTERPRETATION_PROMPT.format(
        cluster_profiles=json.dumps(state["cluster_profiles"], indent=2, default=str),
        metrics=json.dumps(state["metrics"], indent=2, default=str),
        n_clusters=state["n_clusters"],
        original_columns=json.dumps(state.get("original_columns", [])),
        context=context,
    )

    response = llm.invoke(prompt)
    raw = json.loads(extract_json(response.content))
    if "archetypes" in raw:
        raw["archetypes"] = [a for a in raw["archetypes"] if isinstance(a, dict)]
    result = InterpretationResult(**raw)

    archetypes = [a.model_dump() for a in result.archetypes]

    return {
        "archetypes": archetypes,
        "log_messages": [
            f"[interpret] Generated {len(archetypes)} archetype descriptions",
            f"[interpret] Summary: {result.summary[:100]}..."
            if len(result.summary) > 100
            else f"[interpret] Summary: {result.summary}",
        ],
    }
