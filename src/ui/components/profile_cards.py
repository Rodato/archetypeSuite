import streamlit as st


def render_profile_cards(profile: dict):
    st.subheader("Detalle de Columnas")

    for col_info in profile["columns"]:
        with st.expander(f"**{col_info['name']}** ({col_info['dtype']})"):
            col1, col2, col3 = st.columns(3)
            col1.metric("Valores únicos", col_info["n_unique"])
            col2.metric("Faltantes", col_info["n_missing"])
            col3.metric("% Faltantes", f"{col_info['pct_missing']:.1%}")

            if col_info["is_numeric"]:
                st.markdown("**Estadísticas Numéricas**")
                ncol1, ncol2, ncol3, ncol4 = st.columns(4)
                ncol1.metric("Media", f"{col_info.get('mean', 0):.2f}")
                ncol2.metric("Desv. Est.", f"{col_info.get('std', 0):.2f}")
                ncol3.metric("Mín", f"{col_info.get('min', 0):.2f}")
                ncol4.metric("Máx", f"{col_info.get('max', 0):.2f}")

                ncol5, ncol6, ncol7 = st.columns(3)
                ncol5.metric("Mediana", f"{col_info.get('median', 0):.2f}")
                ncol6.metric("Q1", f"{col_info.get('q1', 0):.2f}")
                ncol7.metric("Q3", f"{col_info.get('q3', 0):.2f}")

            if "top_categories" in col_info:
                st.markdown("**Categorías Principales**")
                st.json(col_info["top_categories"])

    if profile.get("correlation_matrix"):
        st.subheader("Matriz de Correlación")
        import pandas as pd
        import plotly.express as px
        corr_df = pd.DataFrame(profile["correlation_matrix"])
        fig = px.imshow(corr_df, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(height=500)
        st.plotly_chart(fig)
