from src.config.settings import settings
from src.models.state import PipelineState


def should_refine(state: PipelineState) -> str:
    if (
        state.get("should_refine", False)
        and state.get("refinement_count", 0) <= settings.max_refinement_iterations
    ):
        return "cluster"
    return "end"
