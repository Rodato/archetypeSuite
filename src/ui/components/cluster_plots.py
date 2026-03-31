from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_metrics_summary(metrics: dict):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Clusters", metrics.get("n_clusters", "N/A"))
    sil = metrics.get("silhouette_score")
    col2.metric("Silhouette", f"{sil:.3f}" if sil is not None else "N/A")
    ch = metrics.get("calinski_harabasz_score")
    col3.metric("Calinski-Harabasz", f"{ch:.1f}" if ch is not None else "N/A")
    db = metrics.get("davies_bouldin_score")
    col4.metric("Davies-Bouldin", f"{db:.3f}" if db is not None else "N/A")

    sizes = metrics.get("cluster_sizes", {})
    if sizes:
        fig = px.bar(
            x=[str(k) for k in sizes.keys()],
            y=list(sizes.values()),
            labels={"x": "Cluster", "y": "Tamaño"},
            title="Tamaño de Clusters",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_scatter_2d(scatter_df: pd.DataFrame):
    fig = px.scatter(
        scatter_df,
        x="PC1",
        y="PC2",
        color="Arquetipo",
        title="Clusters en 2D (proyección PCA)",
        hover_data=["Cluster"],
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)


def render_radar_chart(
    cluster_profiles: dict,
    numeric_cols: list[str],
    label_map: dict[int, str],
):
    fig = go.Figure()

    all_means: dict[str, list[float]] = {col: [] for col in numeric_cols}
    for col in numeric_cols:
        for cluster_id in sorted(cluster_profiles.keys(), key=int):
            profile = cluster_profiles[str(cluster_id)] if str(cluster_id) in cluster_profiles else cluster_profiles.get(int(cluster_id), {})
            val = profile.get(col, {}).get("mean", 0) or 0
            all_means[col].append(val)

    # Normalize per column for radar display
    for col in numeric_cols:
        vals = all_means[col]
        min_v, max_v = min(vals), max(vals)
        rng = max_v - min_v if max_v != min_v else 1.0
        all_means[col] = [(v - min_v) / rng for v in vals]

    for idx, cluster_id in enumerate(sorted(cluster_profiles.keys(), key=lambda x: int(x))):
        label = label_map.get(int(cluster_id), f"Cluster {cluster_id}")
        values = [all_means[col][idx] for col in numeric_cols]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=numeric_cols + [numeric_cols[0]],
            fill="toself",
            name=label,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Radar de Arquetipos (normalizado)",
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_box_plots(df: pd.DataFrame, column: str, label_map: dict[int, str]):
    fig = px.box(
        df,
        x="Arquetipo",
        y=column,
        color="Arquetipo",
        title=f"Distribución de {column} por Arquetipo",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
