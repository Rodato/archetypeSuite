import streamlit as st

from src.data.profiler import DataProfiler
from src.ui.components.profile_cards import render_profile_cards


def render():
    st.header("Perfil de Datos")

    if "raw_df" not in st.session_state:
        st.warning("Por favor, carga los datos primero.")
        return

    df = st.session_state["raw_df"]

    if "profile" not in st.session_state or st.button("Actualizar perfil"):
        profiler = DataProfiler()
        st.session_state["profile"] = profiler.profile(df)

    profile = st.session_state["profile"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Filas", profile["n_rows"])
    col2.metric("Columnas", profile["n_cols"])
    col3.metric("Numéricas / Categóricas",
                f"{len(profile['numeric_columns'])} / {len(profile['categorical_columns'])}")

    render_profile_cards(profile)
