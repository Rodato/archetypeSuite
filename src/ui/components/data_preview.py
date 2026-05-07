import pandas as pd
import streamlit as st

from src.config.constants import MAX_PREVIEW_ROWS


def render_data_preview(df: pd.DataFrame):
    st.dataframe(df.head(MAX_PREVIEW_ROWS), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Tipos de datos**")
        st.dataframe(
            df.dtypes.astype(str).reset_index().rename(
                columns={"index": "Columna", 0: "Tipo"}
            ),
            use_container_width=True,
        )
    with col2:
        st.markdown("**Valores faltantes**")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            st.success("Sin valores faltantes")
        else:
            st.dataframe(
                missing.reset_index().rename(
                    columns={"index": "Columna", 0: "Faltantes"}
                ),
                use_container_width=True,
            )
