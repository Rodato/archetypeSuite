import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

from src.ui.components.archetype_cards import render_archetype_cards
from src.ui.components.cluster_plots import (
    render_box_plots,
    render_cluster_sizes,
    render_metrics_summary,
    render_quality_card,
    render_radar_chart,
    render_scatter_2d,
)
from src.ui.export import archetypes_to_csv, build_markdown_report, labels_to_csv


def render():
    advanced = st.session_state.get("advanced_mode", False)

    st.header("3. Arquetipos")

    if "pipeline_result" not in st.session_state:
        st.info(
            "Aún no has ejecutado el análisis. Ve al paso **2. Analizar** "
            "para descubrir los arquetipos en tus datos."
        )
        if "raw_df" not in st.session_state:
            st.caption("Primero carga un dataset en **1. Datos**.")
        return

    result = st.session_state["pipeline_result"]
    metrics = result.get("metrics", {})
    archetypes = result.get("archetypes", [])
    labels = result.get("labels", [])
    raw_data = result.get("raw_data", {})
    processed_data = result.get("processed_data", {})
    cluster_profiles = result.get("cluster_profiles", {})

    render_quality_card(metrics)

    label_map = {a["cluster_id"]: a["label"] for a in archetypes}
    render_cluster_sizes(metrics, label_map)

    st.divider()
    st.subheader("Tus arquetipos")
    render_archetype_cards(archetypes)

    if archetypes:
        st.subheader("Descargar resultados")
        col1, col2, col3 = st.columns(3)
        col1.download_button(
            "Arquetipos (CSV)",
            data=archetypes_to_csv(archetypes),
            file_name="arquetipos.csv",
            mime="text/csv",
        )
        if raw_data and labels:
            col2.download_button(
                "Datos etiquetados (CSV)",
                data=labels_to_csv(raw_data, labels, archetypes),
                file_name="datos_etiquetados.csv",
                mime="text/csv",
            )
        col3.download_button(
            "Reporte (Markdown)",
            data=build_markdown_report(result).encode("utf-8"),
            file_name="reporte_arquetipos.md",
            mime="text/markdown",
        )

    st.divider()
    st.subheader("Explorar visualmente")

    raw_df = pd.DataFrame(raw_data)
    if labels:
        raw_df["Cluster"] = labels
        raw_df["Arquetipo"] = raw_df["Cluster"].map(label_map).fillna("Desconocido")

    tab1, tab2, tab3 = st.tabs(["🗺️ Mapa", "📊 Comparar", "📈 Por variable"])

    with tab1:
        st.caption(
            "Cada punto es una fila de tu dataset, coloreada por arquetipo. "
            "Las filas parecidas entre sí aparecen cerca."
        )
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
                scatter_df["Arquetipo"] = (
                    scatter_df["Cluster"].map(label_map).fillna("Desconocido")
                )
                render_scatter_2d(scatter_df)

    with tab2:
        st.caption(
            "Compara arquetipos en varias variables a la vez. "
            "Los valores están normalizados para que se puedan comparar directamente."
        )
        numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "Cluster"]
        if cluster_profiles and numeric_cols:
            render_radar_chart(cluster_profiles, numeric_cols[:8], label_map)

    with tab3:
        st.caption(
            "Elige una variable para ver cómo se distribuye entre los arquetipos."
        )
        numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != "Cluster"]
        if numeric_cols:
            selected_col = st.selectbox("Variable", numeric_cols)
            render_box_plots(raw_df, selected_col, label_map)

    if advanced:
        st.divider()
        st.subheader("Detalles técnicos")
        with st.expander("Métricas de clustering"):
            render_metrics_summary(metrics)
        if result.get("selection_reasoning"):
            with st.expander("Razonamiento de selección de algoritmo"):
                st.markdown(result["selection_reasoning"])
        if result.get("refinement_reason"):
            with st.expander("Decisión de refinamiento"):
                st.markdown(result["refinement_reason"])
        logs = st.session_state.get("pipeline_logs", [])
        if logs:
            with st.expander("Logs del pipeline"):
                st.code("\n".join(logs), language="text")
