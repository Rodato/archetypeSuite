"""Perfilado de grupos a demanda: hipótesis comportamental Plural para un segmento
definido por el usuario en lenguaje natural — sin depender del clustering.

Flujo: NL → filtros (LLM json) → subset determinista → stats grupo-vs-total →
narrativa (LLM narrativo + metodología) con piso de cautela por tamaño de muestra.
"""
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from src.core.quality import CAUTION_ORDER
from src.llm.data_qa import _apply_filters, _summarize_columns
from src.llm.methodology import load_methodology
from src.llm.prompts import GROUP_FILTER_PROMPT, GROUP_PROFILE_PROMPT
from src.llm.provider import get_llm_json, get_narrative_llm, invoke_json_with_retry
from src.models.schemas import GroupFilterSpec, GroupProfileDescription


@dataclass
class GroupProfileResult:
    profile: Optional[Dict[str, Any]] = None
    n: int = 0
    share: float = 0.0
    filters: List[Dict[str, Any]] = field(default_factory=list)
    interpretation: str = ""
    error: Optional[str] = None


def caution_floor_for_group_size(n: int) -> str:
    """Piso determinista: un grupo chico no admite hipótesis confiadas."""
    if n < 30:
        return "alta"
    if n < 100:
        return "media"
    return "baja"


def _group_stats(df: pd.DataFrame, group: pd.DataFrame, max_numeric: int = 10, max_cat: int = 8) -> str:
    """Evidencia compacta grupo-vs-total para el prompt (medias y top categorías)."""
    lines: List[str] = []
    numeric = df.select_dtypes(include="number").columns.tolist()[:max_numeric]
    for col in numeric:
        gm, tm = group[col].mean(), df[col].mean()
        if pd.notna(gm) and pd.notna(tm):
            lines.append(f"- {col}: grupo {gm:.2f} · total {tm:.2f}")
    cats = [c for c in df.columns if c not in numeric and c != "Cluster"][:max_cat]
    for col in cats:
        top = group[col].value_counts(normalize=True).head(3)
        total = df[col].value_counts(normalize=True)
        parts = [
            f"{idx}: {share * 100:.0f}% (total {total.get(idx, 0) * 100:.0f}%)"
            for idx, share in top.items()
        ]
        if parts:
            lines.append(f"- {col}: " + " · ".join(parts))
    return "\n".join(lines) if lines else "(sin estadísticas disponibles)"


def profile_group(df: pd.DataFrame, group_description: str, *, context: str = "") -> GroupProfileResult:
    # 1) NL → filtros ejecutables
    filter_prompt = GROUP_FILTER_PROMPT.format(
        columns_summary=json.dumps(_summarize_columns(df), indent=2, default=str, ensure_ascii=False),
        group_description=group_description,
    )
    spec, error = invoke_json_with_retry(
        get_llm_json(),
        filter_prompt,
        GroupFilterSpec,
        lambda: GroupFilterSpec(feasible=False, reason="No pude interpretar la descripción del grupo."),
    )
    if error or not spec.feasible:
        return GroupProfileResult(
            error=spec.reason or "No pude traducir ese grupo a filtros sobre tus columnas.",
        )
    if not spec.filter_by:
        return GroupProfileResult(
            error="La descripción no define criterios que existan en tus datos. "
                  "Intenta nombrar variables o valores concretos.",
        )

    # 2) Subset determinista
    group = _apply_filters(df, spec)
    n = int(len(group))
    filters_dump = [f.model_dump() for f in spec.filter_by]
    if n == 0:
        return GroupProfileResult(
            filters=filters_dump,
            interpretation=spec.interpretation,
            error="Ninguna fila cumple esos criterios. Ajusta la descripción del grupo.",
        )
    share = round(n / len(df) * 100, 1) if len(df) else 0.0

    # 3) Narrativa con metodología + piso de cautela por tamaño de muestra
    profile_prompt = GROUP_PROFILE_PROMPT.format(
        methodology=load_methodology(),
        group_description=group_description,
        interpretation=spec.interpretation or "(literal)",
        n=n,
        share=share,
        stats=_group_stats(df, group),
        context=context or "No se proporcionó contexto adicional.",
    )
    fallback = lambda: GroupProfileDescription(  # noqa: E731
        label=group_description[:60],
        description="No pude generar la narrativa (falló el modelo). "
                    "El grupo existe y sus estadísticas son válidas.",
        nivel_cautela="alta",
        cautela_reason="Narrativa no disponible: tratar como borrador.",
    )
    desc, _narr_error = invoke_json_with_retry(
        get_narrative_llm(), profile_prompt, GroupProfileDescription, fallback,
    )

    floor = caution_floor_for_group_size(n)
    if CAUTION_ORDER.get(desc.nivel_cautela, 0) < CAUTION_ORDER[floor]:
        desc.nivel_cautela = floor
        suffix = f"Grupo de {n} filas: cautela mínima '{floor}' por tamaño de muestra."
        desc.cautela_reason = f"{desc.cautela_reason} {suffix}".strip()

    return GroupProfileResult(
        profile=desc.model_dump(),
        n=n,
        share=share,
        filters=filters_dump,
        interpretation=spec.interpretation,
    )
