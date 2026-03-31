import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

from src.ui.components.cluster_plots import (
    render_box_plots,
    render_radar_chart,
    render_scatter_2d,
)


def render():
    st.header("Explorar Arquetipos")

    if "pipeline_result" not in st.session_state:
        st.warning("Por favor, ejecuta el pipeline primero.")
        return

    result = st.session_state["pipeline_result"]
    labels = result.get("labels", [])
    raw_df = pd.DataFrame(result.get("raw_data", {}))
    processed_data = result.get("processed_data", {})
    archetypes = result.get("archetypes", [])
    cluster_profiles = result.get("cluster_profiles", {})

    raw_df["Cluster"] = labels
    label_map = {a["cluster_id"]: a["label"] for a in archetypes}
    raw_df["Arquetipo"] = raw_df["Cluster"].map(label_map).fillna("Desconocido")

    tab1, tab2, tab3 = st.tabs(["Dispersión", "Radar", "Boxplots"])

    with tab1:
        if processed_data:
            proc_df = pd.DataFrame(processed_data)
            if proc_df.shape[1] >= 2:
                if proc_df.shape[1] > 2:
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(proc_df.values)
                    scatter_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
                else:
                    scatter_df = proc_df.copy()
                    scatter_df.columns = ["PC1", "PC2"]
                scatter_df["Cluster"] = labels
                scatter_df["Arquetipo"] = scatter_df["Cluster"].map(label_map).fillna("Desconocido")
                render_scatter_2d(scatter_df)

    with tab2:
        numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "Cluster"]
        if cluster_profiles and numeric_cols:
            render_radar_chart(cluster_profiles, numeric_cols[:8], label_map)

    with tab3:
        numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "Cluster"]
        if numeric_cols:
            selected_col = st.selectbox("Seleccionar columna", numeric_cols)
            render_box_plots(raw_df, selected_col, label_map)
