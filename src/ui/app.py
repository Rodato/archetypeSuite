import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st

st.set_page_config(
    page_title="Archetype Suite",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
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


def _step_label(step: str, status: str) -> str:
    if status == "done":
        return f"{step} ✓"
    return step


def _resolve_step() -> str:
    forced = st.session_state.pop("_force_page", None)
    if forced and forced in STEPS:
        st.session_state["_current_step"] = forced
        return forced

    current = st.session_state.get("_current_step")
    if current is None:
        if "pipeline_result" in st.session_state:
            current = "Arquetipos"
        elif "raw_df" in st.session_state:
            current = "Analizar"
        else:
            current = "Datos"
        st.session_state["_current_step"] = current

    return current


def _render_topbar() -> str:
    L, C, R = st.columns([1.6, 4, 1.4])

    with L:
        st.markdown(
            "<div class='topbar-brand'>"
            "<span class='topbar-brand__mark'>◆</span>"
            "<span class='topbar-brand__name'>"
            "<strong>Archetype</strong> Suite"
            "</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        if "raw_df" in st.session_state:
            df = st.session_state["raw_df"]
            file_name = st.session_state.get("file_name", "dataset")
            st.markdown(
                f"<div class='dataset-chip'>"
                f"<div class='name'>{file_name}</div>"
                f"<div class='meta'>{df.shape[0]} filas · {df.shape[1]} columnas</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    statuses = {step: _step_status(step) for step in STEPS}

    with C:
        current_step = next((s for s in STEPS if statuses[s] == "current"), STEPS[0])
        current_idx = STEPS.index(current_step) + 1
        st.markdown(
            f"<div class='wizard-progress__label'>Paso {current_idx} de {len(STEPS)}</div>"
            f"<div class='wizard-progress__current-name'>{current_step}</div>"
            "<div class='wizard-progress'>"
            + "".join(
                f"<div class='wizard-progress__step wizard-progress__step--{statuses[s]}'></div>"
                for s in STEPS
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        nav_cols = st.columns(len(STEPS))
        for col, step in zip(nav_cols, STEPS):
            status = statuses[step]
            with col:
                if st.button(
                    _step_label(step, status),
                    key=f"pill_{step}",
                    use_container_width=True,
                    disabled=(status == "locked"),
                    type="primary" if status == "current" else "secondary",
                ):
                    st.session_state["_force_page"] = step
                    st.session_state.pop("_just_finished", None)
                    st.rerun()

    with R:
        st.session_state["advanced_mode"] = st.toggle(
            "Avanzado",
            value=st.session_state.get("advanced_mode", False),
            help="Muestra métricas estadísticas, logs y razonamientos internos.",
        )

    st.markdown("<hr class='topbar-divider'>", unsafe_allow_html=True)
    return _resolve_step()


def main():
    step = _render_topbar()

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
