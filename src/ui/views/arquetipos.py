import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA

from src.ui.components.archetype_cards import render_archetype_cards
from src.ui.components.data_chat import render_data_chat
from src.ui.components.cluster_plots import (
    render_box_plots,
    render_cluster_sizes,
    render_metrics_summary,
    render_quality_card,
    render_radar_chart,
    render_scatter_2d,
    render_silhouette_curve,
)
from src.ui.export import archetypes_to_csv, build_markdown_report, labels_to_csv

ARCHETYPE_TOOLTIP = (
    "**Un arquetipo es un patrón de comportamiento.** "
    "Agrupa a las personas que se parecen entre sí en cómo actúan, "
    "deciden o consumen — no simplemente por edad, género o región. "
    "Cada arquetipo te da un retrato narrativo: cómo es esa persona, "
    "qué la motiva, qué la frena. Sirve para diseñar productos, "
    "campañas o servicios pensando en grupos reales en vez de en un "
    "promedio que no representa a nadie."
)


def render():
    advanced = st.session_state.get("advanced_mode", False)

    if "pipeline_result" not in st.session_state:
        with st.container(border=True):
            st.markdown('<div class="panel--ghost"></div>', unsafe_allow_html=True)
            no_data = "raw_df" not in st.session_state
            st.markdown(
                '<div class="empty-state">'
                '<div class="empty-icon">3</div>'
                f'<div class="empty-title">{"Primero carga un dataset" if no_data else "Aún no has ejecutado el análisis"}</div>'
                f'<div class="empty-desc">{"Ve al paso Datos para cargar tu archivo." if no_data else "Ve al paso Analizar para descubrir los arquetipos en tus datos."}</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            target = "Datos" if no_data else "Analizar"
            if st.button(f"Ir a {target}", type="primary"):
                st.session_state["_force_page"] = target
                st.rerun()
        return

    result = st.session_state["pipeline_result"]
    metrics = result.get("metrics", {})
    archetypes = result.get("archetypes", [])
    labels = result.get("labels", [])
    raw_data = result.get("raw_data", {})
    processed_data = result.get("processed_data", {})
    cluster_profiles = result.get("cluster_profiles", {})

    label_map = {a["cluster_id"]: a["label"] for a in archetypes}
    cluster_sizes_raw = metrics.get("cluster_sizes", {})
    cluster_sizes = {}
    for k, v in cluster_sizes_raw.items():
        try:
            cluster_sizes[int(k)] = int(v)
        except (TypeError, ValueError):
            continue

    k_analysis = result.get("k_analysis")
    optimal_k = result.get("optimal_k") or result.get("n_clusters")

    # Row A: Quality hero panel
    with st.container(border=True):
        hdr_col, info_col = st.columns([10, 1])
        with hdr_col:
            st.markdown('<div class="panel-eyebrow">Resultados</div>', unsafe_allow_html=True)
        with info_col:
            with st.popover("?"):
                st.markdown(ARCHETYPE_TOOLTIP)

        q_col, chart_col = st.columns([2, 3], gap="medium")
        with q_col:
            render_quality_card(metrics)
        with chart_col:
            render_cluster_sizes(metrics, label_map)

    # Row B: Archetype cards
    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="panel-eyebrow">Arquetipos</div>', unsafe_allow_html=True)
        render_archetype_cards(archetypes, cluster_sizes=cluster_sizes)

    # Row C: Explore (large) + Downloads & Why-k (narrow)
    st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
    raw_df = pd.DataFrame(raw_data)
    if labels:
        raw_df["Cluster"] = labels
        raw_df["Arquetipo"] = raw_df["Cluster"].map(label_map).fillna("Desconocido")

    explore_col, side_col = st.columns([3, 1], gap="medium")

    with explore_col:
        with st.container(border=True):
            st.markdown('<div class="panel-eyebrow">Explorar</div>', unsafe_allow_html=True)
            tab1, tab2, tab3, tab4 = st.tabs([
                "Mapa", "Comparar", "Por variable", "Consultar",
            ])

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
                    "Los valores están normalizados para poder compararse directamente."
                )
                numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
                numeric_cols = [c for c in numeric_cols if c != "Cluster"]
                if cluster_profiles and numeric_cols:
                    render_radar_chart(cluster_profiles, numeric_cols[:8], label_map)

            with tab3:
                st.caption("Elige una variable para ver cómo se distribuye entre los arquetipos.")
                numeric_cols = raw_df.select_dtypes(include=np.number).columns.tolist()
                numeric_cols = [c for c in numeric_cols if c != "Cluster"]
                if numeric_cols:
                    selected_col = st.selectbox("Variable", numeric_cols)
                    render_box_plots(raw_df, selected_col, label_map)

            with tab4:
                st.caption(
                    "Haz preguntas como: '¿qué arquetipo tiene la edad promedio más alta?', "
                    "'compara ingreso entre arquetipos', '¿cuántos hay por género en cada arquetipo?'."
                )
                suggestions: list = []
                if archetypes:
                    sample_archetype = archetypes[0].get("label", f"Arquetipo {archetypes[0].get('cluster_id', 0)}")
                    num_in_chat = [c for c in raw_df.select_dtypes(include=np.number).columns if c != "Cluster"]
                    cat_in_chat = [c for c in raw_df.columns if c not in num_in_chat and c not in ("Cluster", "Arquetipo")]
                    if num_in_chat:
                        suggestions.append(f"{num_in_chat[0]} promedio por arquetipo")
                    if cat_in_chat:
                        suggestions.append(f"distribución de {cat_in_chat[0]} en cada arquetipo")
                    suggestions.append(f"¿qué hace único a {sample_archetype}?")
                render_data_chat(
                    raw_df,
                    context=st.session_state.get("dataset_context", ""),
                    mode="archetypes",
                    key="step3",
                    suggestions=suggestions,
                )

    with side_col:
        with st.container(border=True):
            st.markdown('<div class="panel-eyebrow">Descargas</div>', unsafe_allow_html=True)
            if archetypes:
                st.download_button(
                    "Arquetipos (CSV)",
                    data=archetypes_to_csv(archetypes),
                    file_name="arquetipos.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
                if raw_data and labels:
                    st.download_button(
                        "Datos etiquetados (CSV)",
                        data=labels_to_csv(raw_data, labels, archetypes),
                        file_name="datos_etiquetados.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                st.download_button(
                    "Reporte (Markdown)",
                    data=build_markdown_report(result).encode("utf-8"),
                    file_name="reporte_arquetipos.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        if k_analysis and optimal_k:
            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown('<div class="panel-eyebrow">Metodología</div>', unsafe_allow_html=True)
                with st.expander(f"¿Por qué {optimal_k} arquetipos?"):
                    st.markdown(
                        "El sistema probó distintos números de grupos y eligió el que "
                        "produce mejor separación entre ellos."
                    )
                    render_silhouette_curve(k_analysis, int(optimal_k))

    # Advanced: technical details
    if advanced:
        st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
        a1, a2, a3, a4 = st.columns(4, gap="medium")
        with a1:
            with st.container(border=True):
                st.markdown('<div class="panel-eyebrow">Métricas</div>', unsafe_allow_html=True)
                render_metrics_summary(metrics)
        with a2:
            with st.container(border=True):
                st.markdown('<div class="panel-eyebrow">Selección</div>', unsafe_allow_html=True)
                if result.get("selection_reasoning"):
                    with st.expander("Razonamiento"):
                        st.markdown(result["selection_reasoning"])
                else:
                    st.caption("Sin razonamiento disponible.")
        with a3:
            with st.container(border=True):
                st.markdown('<div class="panel-eyebrow">Refinamiento</div>', unsafe_allow_html=True)
                if result.get("refinement_reason"):
                    with st.expander("Decisión"):
                        st.markdown(result["refinement_reason"])
                else:
                    st.caption("Sin refinamiento.")
        with a4:
            with st.container(border=True):
                st.markdown('<div class="panel-eyebrow">Logs</div>', unsafe_allow_html=True)
                logs = st.session_state.get("pipeline_logs", [])
                if logs:
                    with st.expander("Ver logs"):
                        st.code("\n".join(logs), language="text")
                else:
                    st.caption("Sin logs.")
