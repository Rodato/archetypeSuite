"""UI component for the column selection step.

Renders the static-filter summary and the LLM recommendation, and lets the
user toggle which columns to use. Persists the final list in
`st.session_state["selected_columns"]`.
"""
from typing import Any, Dict, List

import streamlit as st


_IMPORTANCE_BADGE = {
    "high": ("🟢", "Alta"),
    "medium": ("🟡", "Media"),
    "low": ("⚪", "Baja"),
}


def _render_static_summary(static_report: Dict[str, Any]) -> None:
    dropped = static_report.get("dropped", [])
    extracted = static_report.get("datetime_extracted", [])
    if not dropped and not extracted:
        st.caption("Ninguna columna fue descartada por filtros automáticos.")
        return

    with st.expander(f"Columnas filtradas automáticamente ({len(dropped)} quitadas, {len(extracted)} fechas convertidas)"):
        if dropped:
            st.markdown("**Quitadas:**")
            for d in dropped:
                st.markdown(f"- `{d['column']}` — {d['reason']}")
        if extracted:
            st.markdown("**Fechas convertidas a año:**")
            for d in extracted:
                st.markdown(f"- `{d['original']}` → `{d['new']}`")


def render_column_selector(
    static_report: Dict[str, Any],
    recommendation: Dict[str, Any],
    available_columns: List[str],
) -> List[str]:
    """Render the column selection UI and return the user-chosen list."""
    st.subheader("Variables a usar para los arquetipos")
    summary = recommendation.get("summary", "")
    if summary:
        st.caption(summary)

    _render_static_summary(static_report)

    selected_recs = recommendation.get("selected_columns", [])
    excluded_recs = recommendation.get("excluded_columns", [])

    selected_names = {r["name"] for r in selected_recs}
    excluded_names = {r["name"] for r in excluded_recs}
    suggestion_by_name = {r["name"]: r for r in selected_recs}

    user_choice: List[str] = []
    st.markdown("**Recomendadas para clustering** (marca/desmarca según lo que tenga sentido)")
    for col in available_columns:
        if col not in selected_names:
            continue
        rec = suggestion_by_name[col]
        emoji, label = _IMPORTANCE_BADGE.get(rec.get("importance", "medium"), _IMPORTANCE_BADGE["medium"])
        checked = st.checkbox(
            f"{emoji} `{col}` — {rec.get('reason', '')}",
            value=True,
            key=f"col_select_{col}",
            help=f"Importancia: {label}",
        )
        if checked:
            user_choice.append(col)

    others = [c for c in available_columns if c not in selected_names]
    if others:
        with st.expander(f"Mostrar columnas no recomendadas ({len(others)})"):
            for col in others:
                reason = next((e["reason"] for e in excluded_recs if e["name"] == col), "")
                checked = st.checkbox(
                    f"`{col}`" + (f" — {reason}" if reason else ""),
                    value=False,
                    key=f"col_other_{col}",
                )
                if checked:
                    user_choice.append(col)

    if not user_choice:
        st.warning("Selecciona al menos una columna para continuar.")

    return user_choice
