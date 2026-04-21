import streamlit as st

from src.agents.graph import compile_graph
from src.ui.quality import natural_log_message


def _seen_steps_from_logs(logs: list) -> list:
    """Devuelve los mensajes amigables únicos que aparecen en los logs, en orden."""
    seen: list = []
    for raw in logs:
        friendly = natural_log_message(raw)
        if friendly and friendly not in seen:
            seen.append(friendly)
    return seen


def render():
    advanced = st.session_state.get("advanced_mode", False)

    st.header("2. Analizar")

    if "raw_df" not in st.session_state:
        st.info("Primero carga un dataset en el paso **1. Datos**.")
        return

    df = st.session_state["raw_df"]
    context = st.session_state.get("dataset_context", "")

    if "pipeline_result" in st.session_state and not st.session_state.get("_just_finished", False):
        st.success("✅ Análisis completado. Puedes ver los resultados en **3. Arquetipos**.")
        if st.button("Volver a analizar", type="secondary"):
            st.session_state.pop("pipeline_result", None)
            st.session_state.pop("pipeline_logs", None)
            st.rerun()
        if st.button("Ver arquetipos →", type="primary"):
            st.session_state["_force_page"] = "Arquetipos"
            st.rerun()
        return

    st.markdown(
        "Vamos a analizar tu dataset para identificar **grupos de personas con comportamientos similares**. "
        "El sistema detecta automáticamente cuántos grupos hay, los caracteriza y les da nombre."
    )

    if context:
        with st.expander("Contexto que usará el análisis"):
            st.markdown(context)

    if not st.button("Generar arquetipos", type="primary"):
        return

    initial_state = {
        "raw_data": df.to_dict(orient="list"),
        "file_name": st.session_state.get("file_name", "desconocido"),
        "dataset_context": context,
        "refinement_count": 0,
        "log_messages": [],
    }

    graph = compile_graph()

    progress_area = st.empty()
    advanced_log_area = st.empty() if advanced else None
    final_state = None
    all_logs: list = []

    with st.spinner("Analizando..."):
        for state in graph.stream(initial_state, stream_mode="values"):
            final_state = state
            logs = state.get("log_messages", [])
            all_logs = logs

            friendly_steps = _seen_steps_from_logs(logs)
            if friendly_steps:
                progress_area.markdown(
                    "\n".join(f"- {step}" for step in friendly_steps)
                )

            if advanced and advanced_log_area is not None:
                advanced_log_area.code("\n".join(logs), language="text")

    st.session_state["pipeline_result"] = final_state
    st.session_state["pipeline_logs"] = all_logs
    st.session_state["_just_finished"] = True

    n_arch = len(final_state.get("archetypes", [])) if final_state else 0
    st.success(f"🎉 Listo — se encontraron **{n_arch} arquetipos**.")

    if st.button("Ver arquetipos →", type="primary"):
        st.session_state["_force_page"] = "Arquetipos"
        st.session_state["_just_finished"] = False
        st.rerun()

    if advanced:
        st.divider()
        st.subheader("Detalles técnicos")
        st.caption(f"Algoritmo: **{final_state.get('selected_algorithm', 'N/A')}**")
        st.caption(f"Iteraciones de refinamiento: **{final_state.get('refinement_count', 0)}**")
        if final_state.get("selection_reasoning"):
            with st.expander("Razonamiento de selección de algoritmo"):
                st.markdown(final_state["selection_reasoning"])
        if final_state.get("refinement_reason"):
            with st.expander("Decisión de refinamiento"):
                st.markdown(final_state["refinement_reason"])
        with st.expander("Logs del pipeline"):
            st.code("\n".join(all_logs), language="text")
