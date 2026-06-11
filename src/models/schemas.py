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


class ArchetypeDescription(BaseModel):
    """Arquetipo como hipótesis comportamental (8 campos del marco metodológico Plural)."""
    cluster_id: int
    label: str  # 4.1 Nombre provisional (descriptivo, no moralizante)
    description: str  # 4.2 Descripción breve del patrón (2-3 oraciones)
    comportamiento_principal: str = ""  # 4.3 Comportamiento observado / patrón principal
    microcomportamientos: List[str] = Field(default_factory=list)  # 4.4
    barreras: List[str] = Field(default_factory=list)  # 4.5 (idealmente con sub-nivel COM-B)
    habilitadores: List[str] = Field(default_factory=list)  # 4.6
    oportunidades_accion: List[str] = Field(default_factory=list)  # 4.7 (pistas, no soluciones)
    nivel_cautela: Literal["baja", "media", "alta"] = "media"  # 4.8
    cautela_reason: str = ""
    # Legacy (esquema de 4 campos) — se conservan para compat con runs persistidos antiguos.
    key_characteristics: List[str] = Field(default_factory=list)
    differentiators: List[str] = Field(default_factory=list)


class InterpretationResult(BaseModel):
    archetypes: List[ArchetypeDescription]
    summary: str = ""


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
    "missing_values",
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


class GroupFilterSpec(BaseModel):
    """Traducción LLM de una descripción de grupo en lenguaje natural a filtros ejecutables.

    Duck-type compatible con `_apply_filters` de data_qa (expone `filter_by`).
    """
    filter_by: List[FilterCondition] = Field(default_factory=list)
    interpretation: str = ""
    feasible: bool = True
    reason: Optional[str] = None


class GroupProfileDescription(BaseModel):
    """Hipótesis comportamental para un grupo definido por el usuario (no emergente del clustering)."""
    label: str
    description: str = ""
    comportamiento_principal: str = ""
    microcomportamientos: List[str] = Field(default_factory=list)
    barreras: List[str] = Field(default_factory=list)
    habilitadores: List[str] = Field(default_factory=list)
    oportunidades_accion: List[str] = Field(default_factory=list)
    nivel_cautela: Literal["baja", "media", "alta"] = "media"
    cautela_reason: str = ""
