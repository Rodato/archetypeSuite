from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PreprocessingDecision(BaseModel):
    drop_columns: List[str] = Field(default_factory=list)
    imputation: str = "median"
    scaling: str = "standard"
    encoding: str = "onehot"
    dimensionality_reduction: Optional[dict] = None
    reasoning: str = ""


class AlgorithmSelection(BaseModel):
    algorithm: str = "KMeans"
    params: dict = Field(default_factory=dict)
    reasoning: str = ""


class ArchetypeDescription(BaseModel):
    cluster_id: int
    label: str
    description: str
    key_characteristics: List[str] = Field(default_factory=list)
    differentiators: List[str] = Field(default_factory=list)


class InterpretationResult(BaseModel):
    archetypes: List[ArchetypeDescription]
    summary: str = ""


class RefinementDecision(BaseModel):
    should_refine: bool = False
    reason: str = ""
    suggested_algorithm: Optional[str] = None
    suggested_params: Optional[dict] = None
