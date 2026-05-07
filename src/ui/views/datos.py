from pathlib import Path

import pandas as pd
import streamlit as st

from src.agents.nodes.column_selection_node import suggest_columns
from src.data.ingest import DataIngestor
from src.data.profiler import DataProfiler
from src.ui.components.column_selector import render_column_selector
from src.ui.components.data_chat import render_data_chat
from src.ui.components.data_preview import render_data_preview
from src.ui.components.profile_cards import render_profile_cards

DEMO_DATASET_PATH = Path(__file__).resolve().parents[3] / "sample_data" / "customers.csv"
DEMO_CONTEXT = (
    "50 clientes de retail. Queremos entender perfiles de comportamiento de compra "
    "para diseñar campañas y experiencias diferenciadas."
)


def _load_demo_dataset() -> None:
    df = pd.read_csv(DEMO_DATASET_PATH)
    DataIngestor().validate(df)
    st.session_state["raw_df"] = df
    st.session_state["file_name"] = DEMO_DATASET_PATH.name
    st.session_state["dataset_context"] = DEMO_CONTEXT
    for k in list(st.session_state.keys()):
        if k.startswith("_column_suggestion::"):
            del st.session_state[k]


def _render_upload_block(compact: bool = False):
    label = "Sube tu archivo" if compact else "Sube tu archivo (CSV o Excel)"
    uploaded_file = st.file_uploader(label, type=["csv", "xlsx", "xls"])
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
                selected_sheet = st.selectbox("Pestaña a usar:", options=sheet_names)
            else:
                selected_sheet = sheet_names[0]
            df = excel_file.parse(selected_sheet)

        ingestor.validate(df)
        st.session_state["raw_df"] = df
        st.session_state["file_name"] = uploaded_file.name
        for k in list(st.session_state.keys()):
            if k.startswith("_column_suggestion::"):
                del st.session_state[k]
        st.rerun()
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
            st.success(f"Consulta cargada: {df.shape[0]} filas, {df.shape[1]} columnas")
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
        st.caption("Sin valores faltantes.")
    else:
        pct = missing / (n_rows * n_cols) * 100
        st.caption(
            f"{missing} valores faltantes ({pct:.1f}%). El sistema los imputará automáticamente."
        )


def _render_column_selection_section(df: pd.DataFrame, context: str) -> None:
    file_name = st.session_state.get("file_name", "")
    suggestion_key = f"_column_suggestion::{file_name}::{hash(context) & 0xFFFF}"

    if suggestion_key not in st.session_state:
        st.caption(
            "Selecciona las variables más relevantes para el análisis. "
            "Esto evita que columnas como `id`, fechas o texto libre contaminen los resultados."
        )
        if st.button("Sugerir variables", type="secondary", use_container_width=True):
            with st.spinner("Analizando columnas..."):
                result = suggest_columns(df, dataset_context=context)
            st.session_state[suggestion_key] = result
            st.rerun()
        return

    suggestion = st.session_state[suggestion_key]
    static_report = suggestion["static_filter_result"]
    recommendation = suggestion["column_recommendation"]
    filtered_df = suggestion["filtered_df"]

    if suggestion.get("llm_error"):
        st.warning(
            f"No pudimos llamar al modelo ({suggestion['llm_error']}). "
            "Te mostramos las columnas válidas para que elijas tú."
        )

    user_choice = render_column_selector(
        static_report=static_report,
        recommendation=recommendation,
        available_columns=list(filtered_df.columns),
    )
    st.session_state["selected_columns"] = user_choice
    st.session_state["static_filter_result"] = static_report
    st.session_state["column_recommendation"] = recommendation
    st.session_state["filtered_df"] = filtered_df

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Re-sugerir", type="secondary", use_container_width=True):
            st.session_state.pop(suggestion_key, None)
            st.rerun()
    with col_b:
        if st.button(
            "Continuar", type="primary",
            disabled=not user_choice, use_container_width=True,
        ):
            st.session_state["_force_page"] = "Analizar"
            st.rerun()


def render():
    advanced = st.session_state.get("advanced_mode", False)

    if "raw_df" not in st.session_state:
        with st.container(border=True):
            st.markdown('<div class="panel--hero"></div>', unsafe_allow_html=True)
            col_left, col_right = st.columns([3, 2], gap="medium")
            with col_left:
                st.markdown('<div class="panel-eyebrow">Paso 1 de 3</div>', unsafe_allow_html=True)
                st.markdown("## Carga tu dataset")
                st.markdown(
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

                if st.button("Usar dataset de ejemplo", type="secondary"):
                    _load_demo_dataset()
                    st.rerun()
                st.caption("Si es tu primera vez, usa el dataset de ejemplo para ver el flujo completo.")

            with col_right:
                st.markdown('<div class="panel-eyebrow">Contexto (opcional pero recomendado)</div>', unsafe_allow_html=True)
                st.markdown('<div class="panel-title">¿Qué es este dataset?</div>', unsafe_allow_html=True)
                context = st.text_area(
                    "Contexto",
                    value=st.session_state.get("dataset_context", ""),
                    placeholder="Ejemplo: Clientes de una tienda de ropa online. Queremos segmentarlos por comportamiento de compra para personalizar campañas.",
                    height=160,
                    label_visibility="collapsed",
                )
                st.session_state["dataset_context"] = context
                st.caption(
                    "El contexto ayuda al sistema a generar nombres y descripciones más precisas para cada arquetipo."
                )
        return

    # Dataset loaded
    df = st.session_state["raw_df"]
    context = st.session_state.get("dataset_context", "")

    # Row A: Dataset | Contexto | Variables
    cola, colb, colc = st.columns([1.2, 2, 2.2], gap="medium")

    with cola:
        with st.container(border=True):
            st.markdown('<div class="panel-eyebrow">Dataset</div>', unsafe_allow_html=True)
            file_name = st.session_state.get("file_name", "dataset")
            st.markdown(f"**{file_name}**")
            st.caption(f"{df.shape[0]} filas · {df.shape[1]} columnas")
            st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
            if advanced:
                source = st.radio("Fuente", ["Subir archivo", "Conexión SQL"], horizontal=True)
                if source == "Subir archivo":
                    _render_upload_block(compact=True)
                else:
                    _render_sql_block()
            else:
                _render_upload_block(compact=True)
            if st.button("Usar ejemplo", type="secondary", use_container_width=True, key="demo_loaded"):
                _load_demo_dataset()
                st.rerun()

    with colb:
        with st.container(border=True):
            st.markdown('<div class="panel--accent"></div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-eyebrow">Contexto</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">¿Qué es este dataset?</div>', unsafe_allow_html=True)
            context = st.text_area(
                "Contexto",
                value=context,
                placeholder="Ejemplo: Clientes de una tienda de ropa online...",
                height=130,
                label_visibility="collapsed",
            )
            st.session_state["dataset_context"] = context

    with colc:
        with st.container(border=True):
            st.markdown('<div class="panel-eyebrow">Variables a usar</div>', unsafe_allow_html=True)
            _render_column_selection_section(df, context)

    # Row B: Vista previa | Chat
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        with st.container(border=True):
            st.markdown('<div class="panel-eyebrow">Vista previa</div>', unsafe_allow_html=True)
            render_data_preview(df)

    with col2:
        st.markdown('<div class="panel-eyebrow">Pregunta sobre tus datos</div>', unsafe_allow_html=True)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in df.columns if c not in numeric_cols]
        suggestions = []
        if cat_cols:
            suggestions.append(f"¿Cuántos hay por {cat_cols[0]}?")
        if numeric_cols:
            suggestions.append(f"Distribución de {numeric_cols[0]}")
        if numeric_cols and cat_cols:
            suggestions.append(f"{numeric_cols[0]} promedio por {cat_cols[0]}")
        render_data_chat(df, context=context, mode="raw", key="step1", suggestions=suggestions)

    # Row C: Resumen natural
    with st.container(border=True):
        st.markdown('<div class="panel--ghost"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-eyebrow">Resumen</div>', unsafe_allow_html=True)
        _render_natural_summary(df)

    if advanced:
        st.markdown("<div style='height:.75rem'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="panel-eyebrow">Estadísticas detalladas</div>', unsafe_allow_html=True)
            if "profile" not in st.session_state or st.button("Actualizar perfil"):
                profiler = DataProfiler()
                st.session_state["profile"] = profiler.profile(df)
            profile = st.session_state["profile"]
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Filas", profile["n_rows"])
            mc2.metric("Columnas", profile["n_cols"])
            mc3.metric(
                "Numéricas / Categóricas",
                f"{len(profile['numeric_columns'])} / {len(profile['categorical_columns'])}",
            )
            render_profile_cards(profile)
