import pandas as pd
import streamlit as st

from src.data.ingest import DataIngestor
from src.data.profiler import DataProfiler
from src.ui.components.data_preview import render_data_preview
from src.ui.components.profile_cards import render_profile_cards


def _render_upload_block():
    uploaded_file = st.file_uploader(
        "Sube tu archivo (CSV o Excel)", type=["csv", "xlsx", "xls"]
    )
    if uploaded_file is None:
        return

    ingestor = DataIngestor()
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            selected_sheet = None
        else:
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            if len(sheet_names) > 1:
                selected_sheet = st.selectbox(
                    "El archivo tiene varias pestañas. ¿Cuál quieres usar?",
                    options=sheet_names,
                )
            else:
                selected_sheet = sheet_names[0]
            df = excel_file.parse(selected_sheet)

        ingestor.validate(df)
        st.session_state["raw_df"] = df
        st.session_state["file_name"] = uploaded_file.name
        label = (
            f"{uploaded_file.name} — pestaña '{selected_sheet}'"
            if selected_sheet
            else uploaded_file.name
        )
        st.success(
            f"Cargado **{label}**: {df.shape[0]} filas, {df.shape[1]} columnas"
        )
    except ValueError as e:
        st.error(str(e))
    except pd.errors.ParserError as e:
        st.error(
            f"No se pudo leer el archivo: {e}. "
            "Verifica el separador — puede que el CSV use `;` o `\\t` en lugar de `,`."
        )
    except UnicodeDecodeError as e:
        st.error(
            f"Problema de codificación: {e}. "
            "Intenta guardar el archivo como UTF-8 antes de subirlo."
        )
    except Exception as e:
        st.error(f"Error al cargar el archivo: {type(e).__name__}: {e}")


def _render_sql_block():
    conn_string = st.text_input("Cadena de conexión")
    query = st.text_area("Consulta SQL")
    if st.button("Ejecutar consulta") and conn_string and query:
        ingestor = DataIngestor()
        try:
            df = ingestor.load_sql(conn_string, query)
            ingestor.validate(df)
            st.session_state["raw_df"] = df
            st.session_state["file_name"] = "sql_query"
            st.success(
                f"Consulta cargada: {df.shape[0]} filas, {df.shape[1]} columnas"
            )
        except Exception as e:
            st.error(str(e))


def _render_natural_summary(df: pd.DataFrame):
    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    missing = int(df.isnull().sum().sum())

    parts = [f"**{n_rows}** filas con **{n_cols}** columnas."]
    if numeric_cols:
        parts.append(f"**{len(numeric_cols)}** numéricas")
    if categorical_cols:
        parts.append(f"**{len(categorical_cols)}** categóricas")
    summary = " · ".join(parts)

    st.markdown(summary)
    if missing == 0:
        st.caption("✅ Sin valores faltantes.")
    else:
        pct = missing / (n_rows * n_cols) * 100
        st.caption(f"⚠️ {missing} valores faltantes ({pct:.1f}%). El sistema los imputará automáticamente.")


def render():
    advanced = st.session_state.get("advanced_mode", False)

    st.header("1. Datos")
    st.markdown(
        "Sube un dataset para descubrir arquetipos. "
        "Cada fila debe representar una persona, cliente, encuesta o unidad de análisis."
    )

    if advanced:
        source = st.radio("Fuente", ["Subir archivo", "Conexión SQL"], horizontal=True)
        if source == "Subir archivo":
            _render_upload_block()
        else:
            _render_sql_block()
    else:
        _render_upload_block()

    st.subheader("Contexto (opcional pero recomendado)")
    context = st.text_area(
        "Cuéntanos qué representa este dataset: qué es cada fila, el dominio, y para qué lo vas a usar. "
        "Esto ayuda al sistema a generar descripciones más precisas.",
        value=st.session_state.get("dataset_context", ""),
        placeholder="Ejemplo: Clientes de una tienda de ropa online. "
                    "Queremos segmentarlos por comportamiento de compra para personalizar campañas.",
        height=100,
    )
    st.session_state["dataset_context"] = context

    if "raw_df" in st.session_state:
        df = st.session_state["raw_df"]

        st.divider()
        st.subheader("Resumen")
        _render_natural_summary(df)

        render_data_preview(df)

        if advanced:
            st.divider()
            st.subheader("Estadísticas detalladas")
            if "profile" not in st.session_state or st.button("Actualizar perfil"):
                profiler = DataProfiler()
                st.session_state["profile"] = profiler.profile(df)
            profile = st.session_state["profile"]
            col1, col2, col3 = st.columns(3)
            col1.metric("Filas", profile["n_rows"])
            col2.metric("Columnas", profile["n_cols"])
            col3.metric(
                "Numéricas / Categóricas",
                f"{len(profile['numeric_columns'])} / {len(profile['categorical_columns'])}",
            )
            render_profile_cards(profile)

        st.divider()
        if st.button("Continuar al análisis →", type="primary"):
            st.session_state["_force_page"] = "Analizar"
            st.rerun()
