import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config.constants import MAX_PREVIEW_ROWS

# Color tokens (deben ir en sintonía con styles.py — primario, secundario, etc.)
TYPE_COLORS = {
    "Numéricas": "#4F46E5",
    "Categóricas": "#F59E0B",
    "Fechas": "#10B981",
    "Texto libre": "#94A3B8",
}


def render_data_preview(df: pd.DataFrame):
    st.dataframe(df.head(MAX_PREVIEW_ROWS), use_container_width=True, hide_index=True)
    st.caption(f"Primeras {min(MAX_PREVIEW_ROWS, len(df))} filas de {len(df)}.")


def _classify_columns(df: pd.DataFrame) -> dict:
    counts = {"Numéricas": 0, "Categóricas": 0, "Fechas": 0, "Texto libre": 0}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            counts["Numéricas"] += 1
        elif pd.api.types.is_datetime64_any_dtype(s):
            counts["Fechas"] += 1
        else:
            # Texto libre = >50% valores únicos sobre filas no nulas
            non_null = s.dropna()
            if len(non_null) > 0 and non_null.nunique() / len(non_null) > 0.5:
                counts["Texto libre"] += 1
            else:
                counts["Categóricas"] += 1
    return {k: v for k, v in counts.items() if v > 0}


def render_type_donut(df: pd.DataFrame):
    """Pequeño donut con la composición de tipos del dataset + missing %."""
    counts = _classify_columns(df)
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = (df.isnull().sum().sum() / total_cells * 100) if total_cells > 0 else 0.0

    labels = list(counts.keys())
    values = list(counts.values())
    colors = [TYPE_COLORS.get(l, "#CBD5E1") for l in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.62,
                marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
                textinfo="value",
                textfont=dict(family="Inter, sans-serif", size=12, color="#FFFFFF"),
                hovertemplate="<b>%{label}</b><br>%{value} columnas (%{percent})<extra></extra>",
                sort=False,
            )
        ]
    )
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            x=1.05,
            y=0.5,
            xanchor="left",
            yanchor="middle",
            font=dict(family="Inter, sans-serif", size=12, color="#475569"),
        ),
        annotations=[
            dict(
                text=f"<b>{df.shape[1]}</b><br><span style='font-size:0.75rem;color:#64748B'>columnas</span>",
                x=0.5, y=0.5,
                font=dict(family="Inter, sans-serif", size=18, color="#0F172A"),
                showarrow=False,
            )
        ],
        margin=dict(l=0, r=0, t=10, b=10),
        height=180,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if missing_pct > 0:
        st.caption(f"⚠️ {missing_pct:.1f}% de los valores están vacíos (los imputaremos automáticamente).")
    else:
        st.caption("✓ Sin valores faltantes.")
