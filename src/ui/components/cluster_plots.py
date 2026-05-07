from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui.quality import silhouette_to_quality

BRAND_PALETTE = ["#4F46E5", "#F59E0B", "#10B981", "#EF4444", "#8B5CF6", "#EC4899", "#14B8A6", "#F97316"]


def _apply_brand_layout(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=BRAND_PALETTE,
        font=dict(family="Inter, sans-serif", size=12, color="#475569"),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor="rgba(0,0,0,0.06)", zeroline=False),
        yaxis=dict(gridcolor="rgba(0,0,0,0.06)", zeroline=False),
    )
    return fig


def render_quality_card(metrics: dict):
    sil = metrics.get("silhouette_score")
    quality = silhouette_to_quality(sil)
    score_str = f"{sil:.2f}" if sil is not None else "—"

    color_map = {
        "green": {"bg": "#DCFCE7", "text": "#15803D", "border": "#86EFAC"},
        "orange": {"bg": "#FEF9C3", "text": "#B45309", "border": "#FDE68A"},
        "red": {"bg": "#FEE2E2", "text": "#DC2626", "border": "#FCA5A5"},
        "gray": {"bg": "#F1F5F9", "text": "#64748B", "border": "#CBD5E1"},
    }
    c = color_map.get(quality.get("color", "gray"), color_map["gray"])
    grade = quality.get("grade", "—")

    bg = c["bg"]
    text_color = c["text"]
    border_color = c["border"]
    quality_label = quality["label"]
    quality_desc = quality["description"]
    st.markdown(
        f"<div class='quality-hero'>"
        f"<div class='qh-grade' style='background:{bg};color:{text_color};border:2px solid {border_color}'>"
        f"{grade}"
        f"</div>"
        f"<div style='flex:1'>"
        f"<div class='label'>Calidad del análisis</div>"
        f"<div class='value'>{quality_label}</div>"
        f"<div class='desc'>{quality_desc}</div>"
        f"</div>"
        f"<div style='font-size:1.6rem;font-weight:700;color:{text_color};letter-spacing:-0.03em;margin-left:0.5rem'>{score_str}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


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
        _apply_brand_layout(fig)
        st.plotly_chart(fig, use_container_width=True)


def render_cluster_sizes(metrics: dict, label_map: dict[int, str] | None = None):
    sizes = metrics.get("cluster_sizes", {})
    if not sizes:
        return
    keys = list(sizes.keys())
    labels = [
        (label_map or {}).get(int(k), f"Arquetipo {k}") for k in keys
    ]
    fig = px.bar(
        x=labels,
        y=list(sizes.values()),
        labels={"x": "", "y": "Personas"},
    )
    fig.update_traces(marker_color="#4F46E5", marker_line_width=0)
    _apply_brand_layout(fig)
    fig.update_layout(showlegend=False, height=240, title=None)
    st.plotly_chart(fig, use_container_width=True)


def render_scatter_2d(scatter_df: pd.DataFrame):
    fig = px.scatter(
        scatter_df,
        x="PC1",
        y="PC2",
        color="Arquetipo",
        hover_data=["Cluster"],
        color_discrete_sequence=BRAND_PALETTE,
    )
    fig.update_traces(marker=dict(size=9, opacity=0.8, line=dict(width=0)))
    _apply_brand_layout(fig)
    fig.update_layout(
        height=480,
        xaxis_title="Eje 1",
        yaxis_title="Eje 2",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
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

    for col in numeric_cols:
        vals = all_means[col]
        min_v, max_v = min(vals), max(vals)
        rng = max_v - min_v if max_v != min_v else 1.0
        all_means[col] = [(v - min_v) / rng for v in vals]

    for idx, cluster_id in enumerate(sorted(cluster_profiles.keys(), key=lambda x: int(x))):
        label = label_map.get(int(cluster_id), f"Arquetipo {cluster_id}")
        values = [all_means[col][idx] for col in numeric_cols]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=numeric_cols + [numeric_cols[0]],
            fill="toself",
            name=label,
            line=dict(color=BRAND_PALETTE[idx % len(BRAND_PALETTE)]),
        ))

    _apply_brand_layout(fig)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(0,0,0,0.1)"),
            angularaxis=dict(gridcolor="rgba(0,0,0,0.1)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_box_plots(df: pd.DataFrame, column: str, label_map: dict[int, str]):
    fig = px.box(
        df,
        x="Arquetipo",
        y=column,
        color="Arquetipo",
        color_discrete_sequence=BRAND_PALETTE,
    )
    _apply_brand_layout(fig)
    fig.update_layout(height=420, title=None, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def render_silhouette_curve(k_analysis: dict, optimal_k: int) -> None:
    if not k_analysis:
        return
    sil_scores = k_analysis.get("silhouette_scores")
    k_values = k_analysis.get("k_range") or k_analysis.get("k_values")
    if not sil_scores or not k_values:
        return

    k_list = list(k_values)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_list,
        y=list(sil_scores),
        mode="lines+markers",
        line=dict(color="#4F46E5", width=2.5),
        marker=dict(size=8, color="#4F46E5"),
        name="Silhouette",
    ))

    if optimal_k in k_list:
        idx = k_list.index(optimal_k)
        fig.add_trace(go.Scatter(
            x=[optimal_k],
            y=[sil_scores[idx]],
            mode="markers",
            marker=dict(size=16, color="#F59E0B", line=dict(color="#fff", width=2)),
            name=f"k elegido ({optimal_k})",
        ))

    _apply_brand_layout(fig)
    fig.update_layout(
        height=280,
        xaxis_title="Número de arquetipos (k)",
        yaxis_title="Calidad de separación",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
