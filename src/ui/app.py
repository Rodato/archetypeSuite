import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

st.set_page_config(
    page_title="Archetype Suite",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.ui.styles import inject as inject_styles
inject_styles()


STEPS = ["Datos", "Analizar", "Arquetipos"]


def _step_status(step: str) -> str:
    has_data = "raw_df" in st.session_state
    has_result = "pipeline_result" in st.session_state

    if step == "Datos":
        return "done" if has_data else "current"
    if step == "Analizar":
        if not has_data:
            return "locked"
        return "done" if has_result else "current"
    if step == "Arquetipos":
        if not has_result:
            return "locked"
        return "current"
    return "locked"


def _step_icon(status: str) -> str:
    return {"done": "✅", "current": "🔵", "locked": "⚪"}.get(status, "⚪")


def _render_sidebar() -> str:
    st.sidebar.markdown(
        "<div style='padding: 0.5rem 0 0.25rem 0;'>"
        "<div style='font-size: 1.15rem; font-weight: 600; letter-spacing: -0.01em;'>Archetype Suite</div>"
        "<div style='font-size: 0.75rem; color: #6B7280; margin-top: 0.15rem;'>"
        "Segmentación por arquetipos"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    labels = []
    for i, step in enumerate(STEPS, start=1):
        icon = _step_icon(_step_status(step))
        labels.append(f"{icon} {i}. {step}")

    default_index = 0
    forced = st.session_state.pop("_force_page", None)
    if forced and forced in STEPS:
        default_index = STEPS.index(forced)
    elif "pipeline_result" in st.session_state:
        default_index = 2
    elif "raw_df" in st.session_state:
        default_index = 1

    selected_label = st.sidebar.radio(
        "Paso",
        labels,
        index=default_index,
        label_visibility="collapsed",
    )
    selected_step = STEPS[labels.index(selected_label)]

    st.sidebar.divider()

    if "raw_df" in st.session_state:
        df = st.session_state["raw_df"]
        file_name = st.session_state.get("file_name", "dataset")
        st.sidebar.markdown(
            f"<div class='dataset-chip'>"
            f"<div class='name'>{file_name}</div>"
            f"<div class='meta'>{df.shape[0]} filas · {df.shape[1]} columnas</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.session_state["advanced_mode"] = st.sidebar.toggle(
        "⚙️ Modo avanzado",
        value=st.session_state.get("advanced_mode", False),
        help="Muestra métricas estadísticas, logs y razonamientos internos del sistema.",
    )

    return selected_step


def main():
    step = _render_sidebar()

    if step == "Datos":
        from src.ui.views.datos import render
        render()
    elif step == "Analizar":
        from src.ui.views.analizar import render
        render()
    elif step == "Arquetipos":
        from src.ui.views.arquetipos import render
        render()


if __name__ == "__main__":
    main()
