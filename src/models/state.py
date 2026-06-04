import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class PipelineState(TypedDict, total=False):
    # Input
    raw_data: Dict[str, list]
    file_name: str
    dataset_context: Optional[str]

    # Profiling
    profile: Dict[str, Any]

    # Column selection (static filters + LLM relevance)
    static_filter_result: Dict[str, Any]
    column_recommendation: Dict[str, Any]
    selected_columns: List[str]
    datetime_columns: List[str]

    # Preprocessing
    preprocess_strategy: Dict[str, Any]
    processed_data: Dict[str, list]
    original_columns: List[str]

    # K optimization
    optimal_k: int
    k_analysis: Dict[str, Any]

    # Algorithm selection
    selected_algorithm: str
    algorithm_params: Dict[str, Any]
    selection_reasoning: str

    # Clustering results
    labels: List[int]
    n_clusters: int

    # Evaluation
    metrics: Dict[str, Any]
    cluster_profiles: Dict[str, Any]

    # Interpretation
    archetypes: List[Dict[str, Any]]

    # Refinement control
    should_refine: bool
    refinement_reason: str
    refinement_count: int

    # Logging
    log_messages: Annotated[List[str], operator.add]
