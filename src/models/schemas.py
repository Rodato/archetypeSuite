from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


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


FilterOp = Literal["eq", "ne", "gt", "lt", "gte", "lte", "in", "contains"]


class FilterCondition(BaseModel):
    column: str
    op: FilterOp
    value: Union[str, int, float, List[Any]]


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
ChartType = Literal["bar", "pie", "histogram", "box", "scatter", "line", "heatmap", "table", "none"]
AggFunc = Literal["mean", "median", "sum", "min", "max", "count"]
NormalizeMode = Literal["none", "row_pct", "total_pct"]


class BinSpec(BaseModel):
    column: str
    edges: List[float] = Field(default_factory=list)
    labels: Optional[List[str]] = None


class DataQuery(BaseModel):
    operation: DataOperation
    columns: List[str] = Field(default_factory=list)
    groupby: Optional[List[str]] = None
    agg: Optional[AggFunc] = None
    top_n: Optional[int] = None
    bins: Optional[List[BinSpec]] = None
    filter_by: Optional[List[FilterCondition]] = None
    chart_type: ChartType = "table"
    narrative: str = ""
    normalize: NormalizeMode = "none"
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    clarification_options: Optional[List[str]] = None

    @field_validator("columns", mode="before")
    @classmethod
    def _columns_none_to_empty(cls, v):
        return [] if v is None else v
