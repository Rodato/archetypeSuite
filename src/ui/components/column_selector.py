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
        return

    label = []
    if dropped:
        label.append(f"{len(dropped)} quitadas")
    if extracted:
        label.append(f"{len(extracted)} fechas convertidas")
    with st.expander(f"Filtros automáticos previos · {' · '.join(label)}"):
        if dropped:
            st.markdown("**Quitadas (no aportan al clustering):**")
            for d in dropped:
                st.markdown(f"- `{d['column']}` — {d['reason']}")
        if extracted:
            st.markdown("**Fechas convertidas a año:**")
            for d in extracted:
                st.markdown(f"- `{d['original']}` → `{d['new']}`")


def _render_recommended_row(col: str, rec: Dict[str, Any]) -> bool:
    """Una fila por variable recomendada: checkbox + nombre + badge + razón."""
    importance = rec.get("importance", "medium")
    label, badge_class = _IMPORTANCE_CONFIG.get(importance, _IMPORTANCE_CONFIG["medium"])
    reason = rec.get("reason", "")

    chk_col, body_col = st.columns([1, 12], gap="small")
    with chk_col:
        checked = st.checkbox(
            " ",
            value=True,
            key=f"col_select_{col}",
            label_visibility="collapsed",
        )
    with body_col:
        st.markdown(
            f"<div style='line-height:1.35'>"
            f"<code style='font-size:0.85rem'>{col}</code> "
            f"<span class='importance-badge {badge_class}'>{label}</span>"
            f"<div style='color:var(--text-muted);font-size:0.78rem;margin-top:0.15rem'>{reason}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    return checked


def _render_other_row(col: str, reason: str) -> bool:
    """Fila por variable NO recomendada: checkbox apagado + nombre + razón de exclusión."""
    chk_col, body_col = st.columns([1, 12], gap="small")
    with chk_col:
        checked = st.checkbox(
            " ",
            value=False,
            key=f"col_other_{col}",
            label_visibility="collapsed",
        )
    with body_col:
        line = f"<code style='font-size:0.85rem'>{col}</code>"
        if reason:
            line += (
                f"<div style='color:var(--text-muted);font-size:0.78rem;margin-top:0.15rem'>"
                f"La IA no la recomienda: {reason}"
                f"</div>"
            )
        st.markdown(f"<div style='line-height:1.35'>{line}</div>", unsafe_allow_html=True)
    return checked


def render_column_selector(
    static_report: Dict[str, Any],
    recommendation: Dict[str, Any],
    available_columns: List[str],
) -> List[str]:
    summary = recommendation.get("summary", "")
    selected_recs = recommendation.get("selected_columns", [])
    excluded_recs = recommendation.get("excluded_columns", [])

    selected_names = {r["name"] for r in selected_recs}
    suggestion_by_name = {r["name"]: r for r in selected_recs}

    n_recommended = sum(1 for c in available_columns if c in selected_names)
    n_total = len(available_columns)

    # Encabezado: la IA recomendó N de M
    st.markdown(
        f"<div style='font-size:0.875rem;margin-bottom:0.4rem'>"
        f"<span style='color:var(--accent);font-weight:600'>✨ La IA recomendó "
        f"{n_recommended} de {n_total} columnas</span> para el clustering."
        f"</div>",
        unsafe_allow_html=True,
    )
    if summary:
        st.caption(summary)

    _render_static_summary(static_report)

    user_choice: List[str] = []

    # Recomendadas — checked por default. El user puede destildar.
    if n_recommended > 0:
        st.markdown(
            "<div style='font-weight:600;font-size:0.825rem;color:var(--text-secondary);"
            "margin:0.7rem 0 0.35rem;text-transform:uppercase;letter-spacing:0.05em'>"
            "Recomendadas — destílada las que no quieras"
            "</div>",
            unsafe_allow_html=True,
        )
        for col in available_columns:
            if col not in selected_names:
                continue
            if _render_recommended_row(col, suggestion_by_name[col]):
                user_choice.append(col)

    # Otras disponibles — unchecked. El user puede tildar.
    others = [c for c in available_columns if c not in selected_names]
    if others:
        with st.expander(f"Otras disponibles ({len(others)}) — añade las que quieras"):
            for col in others:
                reason = next((e["reason"] for e in excluded_recs if e["name"] == col), "")
                if _render_other_row(col, reason):
                    user_choice.append(col)

    # Contador en vivo
    st.markdown(
        f"<div style='font-size:0.78rem;color:var(--text-muted);margin-top:0.4rem'>"
        f"<strong style='color:var(--text-primary)'>{len(user_choice)}</strong> "
        f"de {n_total} columnas seleccionadas."
        f"</div>",
        unsafe_allow_html=True,
    )

    if not user_choice:
        st.warning("Selecciona al menos una columna para continuar.")

    return user_choice
