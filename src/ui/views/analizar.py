import streamlit as st

from src.agents.graph import compile_graph
from src.ui.quality import natural_log_message


def _seen_steps_from_logs(logs: list) -> list:
    seen: list = []
    for raw in logs:
        friendly = natural_log_message(raw)
        if friendly and friendly not in seen:
            seen.append(friendly)
    return seen


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
            if st.button("Ir a Datos", type="primary"):
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
                if st.button("Volver a analizar", type="secondary", use_container_width=True):
                    st.session_state.pop("pipeline_result", None)
                    st.session_state.pop("pipeline_logs", None)
                    st.session_state["_current_step"] = "Analizar"
                    st.rerun()
            with bcol2:
                if st.button("Ver arquetipos", type="primary", use_container_width=True):
                    st.session_state["_force_page"] = "Arquetipos"
                    st.rerun()
        return

    # Post-run success
    if st.session_state.get("_just_finished", False):
        result = st.session_state.get("pipeline_result", {})
        n_arch = len(result.get("archetypes", [])) if result else 0
        with st.container(border=True):
            st.markdown('<div class="panel--success"></div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-eyebrow">Análisis completado</div>', unsafe_allow_html=True)
            st.markdown(f"### {n_arch} arquetipos encontrados")
            st.markdown(
                "El sistema identificó los patrones de comportamiento en tu dataset. "
                "Pasa al siguiente paso para explorarlos en detalle."
            )
            if st.button("Ver arquetipos", type="primary", use_container_width=True):
                st.session_state["_force_page"] = "Arquetipos"
                st.session_state["_just_finished"] = False
                st.rerun()

        if advanced:
            final_state = st.session_state.get("pipeline_result", {})
            all_logs = st.session_state.get("pipeline_logs", [])
            st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
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

        clicked = st.button("Iniciar análisis", type="primary")

    if not clicked:
        return

    # Running
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

    run_col, status_col = st.columns([2, 1], gap="medium")
    with run_col:
        with st.container(border=True):
            st.markdown('<div class="panel-eyebrow">Progreso</div>', unsafe_allow_html=True)
            progress_area = st.empty()
    with status_col:
        with st.container(border=True):
            st.markdown('<div class="panel--accent"></div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-eyebrow">En curso</div>', unsafe_allow_html=True)
            status_area = st.empty()

    advanced_log_area = None
    if advanced:
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="panel--ghost"></div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-eyebrow">Logs en tiempo real</div>', unsafe_allow_html=True)
            advanced_log_area = st.empty()

    with st.spinner("Analizando..."):
        for state in graph.stream(initial_state, stream_mode="values"):
            final_state = state
            logs = state.get("log_messages", [])
            all_logs = logs

            friendly_steps = _seen_steps_from_logs(logs)
            if friendly_steps:
                last_i = len(friendly_steps) - 1
                item_parts = []
                for i, step in enumerate(friendly_steps):
                    color = "#4F46E5" if i == last_i else "#94A3B8"
                    marker = "›" if i == last_i else "–"
                    item_parts.append(
                        f"<li style='padding:.2rem 0;color:{color};font-size:.875rem'>{marker} {step}</li>"
                    )
                items = "\n".join(item_parts)
                progress_area.markdown(
                    f"<ul style='list-style:none;padding:0;margin:0'>{items}</ul>",
                    unsafe_allow_html=True,
                )
                status_area.markdown(
                    f"<div style='color:#4F46E5;font-size:.875rem;font-weight:500'>{friendly_steps[-1]}</div>",
                    unsafe_allow_html=True,
                )

            if advanced and advanced_log_area is not None:
                advanced_log_area.code("\n".join(logs), language="text")

    st.session_state["pipeline_result"] = final_state
    st.session_state["pipeline_logs"] = all_logs
    st.session_state["_just_finished"] = True
    st.session_state["_current_step"] = "Analizar"
    st.rerun()
