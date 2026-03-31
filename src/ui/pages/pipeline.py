import streamlit as st

from src.agents.graph import compile_graph
from src.ui.components.pipeline_status import render_pipeline_status


def render():
    st.header("Ejecutar Pipeline")

    if "raw_df" not in st.session_state:
        st.warning("Por favor, carga los datos primero.")
        return

    df = st.session_state["raw_df"]

    if st.button("Ejecutar Pipeline de Arquetipos", type="primary"):
        initial_state = {
            "raw_data": df.to_dict(orient="list"),
            "file_name": st.session_state.get("file_name", "desconocido"),
            "dataset_context": st.session_state.get("dataset_context", ""),
            "refinement_count": 0,
            "log_messages": [],
        }

        graph = compile_graph()

        log_area = st.empty()
        all_logs: list = []

        with st.spinner("Pipeline en ejecución..."):
            final_state = None
            for state in graph.stream(initial_state, stream_mode="values"):
                final_state = state
                logs = state.get("log_messages", [])
                if logs:
                    log_area.code("\n".join(logs), language="text")
            all_logs = final_state.get("log_messages", []) if final_state else []

        st.session_state["pipeline_result"] = final_state
        st.session_state["pipeline_logs"] = all_logs

        st.success("¡Pipeline completado!")
        render_pipeline_status(final_state)

    elif "pipeline_result" in st.session_state:
        st.info("Hay una ejecución previa disponible. Puedes ver los resultados o volver a ejecutar.")
        render_pipeline_status(st.session_state["pipeline_result"])
        if "pipeline_logs" in st.session_state:
            with st.expander("Logs del pipeline"):
                st.code("\n".join(st.session_state["pipeline_logs"]), language="text")
