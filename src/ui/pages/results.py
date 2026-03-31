import streamlit as st

from src.ui.components.archetype_cards import render_archetype_cards
from src.ui.components.cluster_plots import render_metrics_summary


def render():
    st.header("Resultados")

    if "pipeline_result" not in st.session_state:
        st.warning("Por favor, ejecuta el pipeline primero.")
        return

    result = st.session_state["pipeline_result"]
    metrics = result.get("metrics", {})
    archetypes = result.get("archetypes", [])

    st.subheader("Métricas de Clustering")
    render_metrics_summary(metrics)

    st.subheader("Descripción de Arquetipos")
    render_archetype_cards(archetypes)
