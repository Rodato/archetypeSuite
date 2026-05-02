from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ColumnRecommendation(BaseModel):
    name: str
    reason: str = ""
    importance: Literal["high", "medium", "low"] = "medium"


class ColumnExclusion(BaseModel):
    name: str
    reason: str = ""


class ColumnRelevanceDecision(BaseModel):
    selected_columns: List[ColumnRecommendation] = Field(default_factory=list)
    excluded_columns: List[ColumnExclusion] = Field(default_factory=list)
    summary: str = ""


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


DataOperation = Literal[
    "describe",
    "groupby_count",
    "groupby_agg",
    "filter_count",
    "distribution",
    "correlation",
    "top_n",
    "value_counts",
]
ChartType = Literal["bar", "pie", "histogram", "box", "scatter", "table", "none"]
AggFunc = Literal["mean", "median", "sum", "min", "max", "count"]


class DataQuery(BaseModel):
    operation: DataOperation
    columns: List[str] = Field(default_factory=list)
    groupby: Optional[List[str]] = None
    agg: Optional[AggFunc] = None
    top_n: Optional[int] = None
    chart_type: ChartType = "table"
    narrative: str = ""
