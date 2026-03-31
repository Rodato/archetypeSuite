import streamlit as st
import pandas as pd

from src.data.ingest import DataIngestor
from src.ui.components.data_preview import render_data_preview


def render():
    st.header("Cargar Datos")

    upload_method = st.radio("Fuente de datos", ["Subir archivo", "Conexión SQL"])

    if upload_method == "Subir archivo":
        uploaded_file = st.file_uploader(
            "Elige un archivo CSV o Excel", type=["csv", "xlsx", "xls"]
        )
        if uploaded_file is not None:
            ingestor = DataIngestor()
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                ingestor.validate(df)
                st.session_state["raw_df"] = df
                st.session_state["file_name"] = uploaded_file.name
                st.success(f"Cargado **{uploaded_file.name}**: {df.shape[0]} filas, {df.shape[1]} columnas")
                render_data_preview(df)
            except ValueError as e:
                st.error(str(e))
    else:
        conn_string = st.text_input("Cadena de conexión")
        query = st.text_area("Consulta SQL")
        if st.button("Ejecutar consulta") and conn_string and query:
            ingestor = DataIngestor()
            try:
                df = ingestor.load_sql(conn_string, query)
                ingestor.validate(df)
                st.session_state["raw_df"] = df
                st.session_state["file_name"] = "sql_query"
                st.success(f"Consulta cargada: {df.shape[0]} filas, {df.shape[1]} columnas")
                render_data_preview(df)
            except Exception as e:
                st.error(str(e))

    st.subheader("Contexto del dataset (opcional)")
    context = st.text_area(
        "Describe el dataset: qué representa cada fila, el dominio, el objetivo del análisis, "
        "columnas clave, etc. Esta información ayuda al pipeline a tomar mejores decisiones.",
        value=st.session_state.get("dataset_context", ""),
        placeholder="Ejemplo: Dataset de clientes de una plataforma de e-commerce. "
                    "Cada fila es un cliente. El objetivo es segmentar por comportamiento de compra "
                    "para personalizar campañas de marketing.",
        height=120,
    )
    st.session_state["dataset_context"] = context

    if "raw_df" in st.session_state:
        st.info(f"Dataset actual: **{st.session_state.get('file_name', 'desconocido')}** "
                f"({st.session_state['raw_df'].shape[0]} filas, "
                f"{st.session_state['raw_df'].shape[1]} columnas)")
