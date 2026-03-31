import streamlit as st


def render_pipeline_status(state: dict):
    col1, col2, col3 = st.columns(3)

    col1.metric("Algoritmo", state.get("selected_algorithm", "N/A"))
    col2.metric("Clusters", state.get("n_clusters", "N/A"))

    refinement_count = state.get("refinement_count", 0)
    col3.metric("Iteraciones de refinamiento", refinement_count)

    metrics = state.get("metrics", {})
    if metrics:
        sil = metrics.get("silhouette_score")
        st.markdown(
            f"**Silhouette:** {sil:.3f}" if sil is not None else "**Silhouette:** N/A"
        )

    if state.get("selection_reasoning"):
        with st.expander("Razonamiento de selección de algoritmo"):
            st.markdown(state["selection_reasoning"])

    if state.get("refinement_reason"):
        with st.expander("Decisión de refinamiento"):
            st.markdown(state["refinement_reason"])
