from typing import Optional

import streamlit as st

from src.agents.graph import compile_graph
from src.config.settings import settings
from src.ui.copy import COPY
from src.ui.quality import PIPELINE_UI_STEPS, nodes_with_logs


def _render_checklist(active_node: Optional[str], completed: set, error_node: Optional[str] = None) -> str:
    """HTML para la lista de pasos esperados con estado visual."""
    items = []
    for key, label in PIPELINE_UI_STEPS:
        if error_node == key:
            cls = "is-failed"
        elif key in completed:
            cls = "is-done"
        elif key == active_node:
            cls = "is-running"
        else:
            cls = "is-pending"
        items.append(
            f"<li class='{cls}'>"
            f"<span class='pipeline-checklist__marker'></span>"
            f"<span>{label}</span>"
            f"</li>"
        )
    return "<ul class='pipeline-checklist'>" + "".join(items) + "</ul>"


def render():
    advanced = st.session_state.get("advanced_mode", False)

    # No data loaded
    if "raw_df" not in st.session_state:
        with st.container(border=True):
            st.markdown('<div class="panel--ghost"></div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="empty-state">'
                '<div class="empty-icon">1</div>'
                '<div class="empty-title">Primero carga un dataset</div>'
                '<div class="empty-desc">Ve al paso Datos y sube un archivo o usa el dataset de ejemplo.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            if st.button(COPY["go_to_data"], type="primary"):
                st.session_state["_force_page"] = "Datos"
                st.rerun()
        return

    df = st.session_state["raw_df"]
    context = st.session_state.get("dataset_context", "")

    # Already completed, returned to this step
    if "pipeline_result" in st.session_state and not st.session_state.get("_just_finished", False):
        with st.container(border=True):
            st.markdown('<div class="panel--success"></div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-eyebrow">Estado</div>', unsafe_allow_html=True)
            st.markdown("### Análisis completado")
            st.markdown("Los arquetipos ya fueron generados. Puedes verlos o volver a ejecutar el análisis.")
            bcol1, bcol2 = st.columns(2)
            with bcol1:
                if st.button(COPY["analyze_again"], type="secondary", use_container_width=True):
                    st.session_state.pop("pipeline_result", None)
                    st.session_state.pop("pipeline_logs", None)
                    st.session_state["_current_step"] = "Analizar"
                    st.rerun()
            with bcol2:
                if st.button(COPY["go_to_results"], type="primary", use_container_width=True):
                    st.session_state["_force_page"] = "Arquetipos"
                    st.rerun()
        return

    # Post-run success
    if st.session_state.get("_just_finished", False):
        result = st.session_state.get("pipeline_result", {})
        n_arch = len(result.get("archetypes", [])) if result else 0
        with st.container(border=True):
            st.markdown('<div class="panel--success"></div>', unsafe_allow_html=True)
            st.markdown(
                "<div class='success-hero'>"
                "<div class='success-hero__sparkle'>✨</div>"
                f"<div class='success-hero__title'>{COPY['analysis_done']}</div>"
                f"<div class='success-hero__sub'>"
                f"Encontramos <strong>{n_arch} arquetipos</strong> en tu dataset. "
                "Pasa al siguiente paso para explorar quiénes son y en qué se diferencian."
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )
            if st.button(COPY["go_to_results"] + " →", type="primary", use_container_width=True):
                st.session_state["_force_page"] = "Arquetipos"
                st.session_state["_just_finished"] = False
                st.rerun()

        if advanced:
            final_state = st.session_state.get("pipeline_result", {})
            all_logs = st.session_state.get("pipeline_logs", [])
            st.markdown("<div class='space-md'></div>", unsafe_allow_html=True)
            adv1, adv2, adv3 = st.columns(3, gap="medium")
            with adv1:
                with st.container(border=True):
                    st.markdown('<div class="panel-eyebrow">Configuración</div>', unsafe_allow_html=True)
                    st.caption(f"Algoritmo: **{final_state.get('selected_algorithm', 'N/A')}**")
                    st.caption(f"Iteraciones de refinamiento: **{final_state.get('refinement_count', 0)}**")
            with adv2:
                with st.container(border=True):
                    st.markdown('<div class="panel-eyebrow">Razonamiento</div>', unsafe_allow_html=True)
                    if final_state.get("selection_reasoning"):
                        with st.expander("Selección de algoritmo"):
                            st.markdown(final_state["selection_reasoning"])
                    if final_state.get("refinement_reason"):
                        with st.expander("Decisión de refinamiento"):
                            st.markdown(final_state["refinement_reason"])
            with adv3:
                with st.container(border=True):
                    st.markdown('<div class="panel-eyebrow">Logs</div>', unsafe_allow_html=True)
                    with st.expander("Ver logs del pipeline"):
                        st.code("\n".join(all_logs), language="text")
        return

    # Pre-run hero
    with st.container(border=True):
        st.markdown('<div class="panel--hero"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-eyebrow">Paso 2 de 3</div>', unsafe_allow_html=True)
        st.markdown("## Generar arquetipos")
        st.markdown(
            "El sistema detecta automáticamente cuántos grupos hay en tu dataset, "
            "los caracteriza y les asigna un nombre basado en patrones de comportamiento."
        )
        if context:
            with st.expander("Contexto que usará el análisis"):
                st.markdown(context)

        clicked = st.button(COPY["analyze_button"], type="primary")

    if not clicked:
        return

    # Validación temprana de API key
    if not settings.openrouter_api_key:
        st.error(COPY["error_no_api_key"])
        st.caption(
            "Crea un archivo `.env` en la raíz del proyecto con la línea "
            "`OPENROUTER_API_KEY=...`. Mira `.env.example` como referencia."
        )
        return

    # Build initial state
    selected_columns = st.session_state.get("selected_columns")
    static_filter_result = st.session_state.get("static_filter_result")
    column_recommendation = st.session_state.get("column_recommendation")
    filtered_df = st.session_state.get("filtered_df")

    if selected_columns and filtered_df is not None:
        df_for_pipeline = filtered_df[selected_columns]
    else:
        df_for_pipeline = df

    initial_state = {
        "raw_data": df_for_pipeline.to_dict(orient="list"),
        "file_name": st.session_state.get("file_name", "desconocido"),
        "dataset_context": context,
        "refinement_count": 0,
        "log_messages": [],
    }
    if selected_columns and static_filter_result and column_recommendation:
        initial_state["selected_columns"] = selected_columns
        initial_state["static_filter_result"] = static_filter_result
        initial_state["column_recommendation"] = column_recommendation

    graph = compile_graph()
    final_state = None
    all_logs: list = []
    last_running_node: Optional[str] = None

    run_col, status_col = st.columns([2, 1], gap="medium")
    with run_col:
        with st.container(border=True):
            st.markdown('<div class="panel-eyebrow">Progreso</div>', unsafe_allow_html=True)
            checklist_area = st.empty()
            checklist_area.markdown(_render_checklist(None, set()), unsafe_allow_html=True)
    with status_col:
        with st.container(border=True):
            st.markdown('<div class="panel--accent"></div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-eyebrow">En curso</div>', unsafe_allow_html=True)
            status_area = st.empty()
            status_area.markdown(
                f"<div style='color:var(--text-muted);font-size:.875rem'>"
                f"Esto suele tomar 1-2 minutos."
                f"</div>",
                unsafe_allow_html=True,
            )

    advanced_log_area = None
    if advanced:
        st.markdown("<div class='space-sm'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="panel--ghost"></div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-eyebrow">Logs en tiempo real</div>', unsafe_allow_html=True)
            advanced_log_area = st.empty()

    pipeline_keys = [k for k, _ in PIPELINE_UI_STEPS]

    try:
        with st.spinner(COPY["analyzing"]):
            for state in graph.stream(initial_state, stream_mode="values"):
                final_state = state
                logs = state.get("log_messages", [])
                all_logs = logs

                completed_nodes = nodes_with_logs(logs) & set(pipeline_keys)
                # Determinar el "siguiente" paso después del último completado
                running = None
                for k in pipeline_keys:
                    if k not in completed_nodes:
                        running = k
                        break
                if running is None and completed_nodes:
                    # All steps already done — show last as running until refinement loop ends
                    running = pipeline_keys[-1]
                last_running_node = running

                checklist_area.markdown(
                    _render_checklist(running, completed_nodes),
                    unsafe_allow_html=True,
                )
                running_label = next(
                    (label for k, label in PIPELINE_UI_STEPS if k == running), "Procesando…"
                )
                status_area.markdown(
                    f"<div style='color:var(--accent);font-size:.875rem;font-weight:500'>"
                    f"{running_label}…"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                if advanced and advanced_log_area is not None:
                    advanced_log_area.code("\n".join(logs), language="text")
    except Exception as e:
        # Marcar el step que estaba corriendo como fallido
        completed_nodes = nodes_with_logs(all_logs) & set(pipeline_keys)
        checklist_area.markdown(
            _render_checklist(None, completed_nodes, error_node=last_running_node),
            unsafe_allow_html=True,
        )
        st.error(COPY["error_pipeline_interrupted"])
        with st.expander("Detalles técnicos"):
            st.code(f"{type(e).__name__}: {e}")
            if all_logs:
                st.caption("Últimos logs antes del error:")
                st.code("\n".join(all_logs[-10:]), language="text")
        if st.button(COPY["retry"], type="primary"):
            st.rerun()
        return

    st.session_state["pipeline_result"] = final_state
    st.session_state["pipeline_logs"] = all_logs
    st.session_state["_just_finished"] = True
    st.session_state["_current_step"] = "Analizar"
    st.rerun()
