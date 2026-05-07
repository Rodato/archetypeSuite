from typing import Any, Dict, List

import streamlit as st


_IMPORTANCE_CONFIG = {
    "high": ("Alta", "importance-badge--high"),
    "medium": ("Media", "importance-badge--medium"),
    "low": ("Baja", "importance-badge--low"),
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
    summary = recommendation.get("summary", "")
    if summary:
        st.caption(summary)

    _render_static_summary(static_report)

    selected_recs = recommendation.get("selected_columns", [])
    excluded_recs = recommendation.get("excluded_columns", [])

    selected_names = {r["name"] for r in selected_recs}
    suggestion_by_name = {r["name"]: r for r in selected_recs}

    user_choice: List[str] = []
    st.markdown("**Recomendadas**")
    for col in available_columns:
        if col not in selected_names:
            continue
        rec = suggestion_by_name[col]
        importance = rec.get("importance", "medium")
        label, badge_class = _IMPORTANCE_CONFIG.get(importance, _IMPORTANCE_CONFIG["medium"])
        badge_html = f"<span class='importance-badge {badge_class}'>{label}</span>"
        checked = st.checkbox(
            f"`{col}` — {rec.get('reason', '')}",
            value=True,
            key=f"col_select_{col}",
            help=f"Importancia: {label}",
        )
        st.markdown(badge_html, unsafe_allow_html=True)
        if checked:
            user_choice.append(col)

    others = [c for c in available_columns if c not in selected_names]
    if others:
        with st.expander(f"Columnas no recomendadas ({len(others)})"):
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
