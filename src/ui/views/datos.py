import pandas as pd
import streamlit as st

from src.agents.nodes.column_selection_node import suggest_columns
from src.data.ingest import DataIngestor
from src.data.profiler import DataProfiler
from src.ui.components.column_selector import render_column_selector
from src.ui.components.data_chat import render_data_chat
from src.ui.components.data_preview import render_data_preview, render_type_donut
from src.ui.components.profile_cards import render_profile_cards
from src.ui.copy import COPY

LOAD_ERROR_MAP = {
    "ParserError": (
        "No pudimos leer el archivo. ¿Está bien formado? "
        "Si es CSV, verifica el separador — puede usar `;` o `\\t` en lugar de `,`."
    ),
    "UnicodeDecodeError": (
        "El archivo usa una codificación que no reconocemos. "
        "Intenta guardarlo como UTF-8 antes de subirlo."
    ),
    "EmptyDataError": "El archivo parece estar vacío.",
    "FileNotFoundError": "No encontramos el archivo. Vuelve a subirlo.",
}


def _humanize_load_error(e: Exception) -> str:
    return LOAD_ERROR_MAP.get(type(e).__name__, COPY["error_load_file"])


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
        # Validación explícita del DataIngestor (filas/columnas mínimas, vacío)
        st.error(str(e))
    except Exception as e:
        st.error(_humanize_load_error(e))
        st.caption(f"Detalle técnico: {type(e).__name__}: {e}")


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
            st.markdown(
                "<div class='hero-onboarding'>"
                "<div class='hero-onboarding__mark'>◆</div>"
                f"<div class='hero-onboarding__title'>{COPY['product_tagline']}</div>"
                "<div class='hero-onboarding__sub'>"
                "Sube un CSV o Excel donde cada fila sea una persona, cliente o unidad de análisis. "
                "El sistema descubre los grupos de comportamiento y los describe en lenguaje natural."
                "</div>"
                "</div>",
                unsafe_allow_html=True,
            )

            _, mid, _ = st.columns([1, 2, 1])
            with mid:
                if advanced:
                    source = st.radio(
                        "Fuente",
                        ["Subir archivo", "Conexión SQL"],
                        horizontal=True,
                        label_visibility="collapsed",
                    )
                    if source == "Subir archivo":
                        _render_upload_block()
                    else:
                        _render_sql_block()
                else:
                    _render_upload_block()
                st.markdown(
                    f"<div class='hero-onboarding__hint' style='text-align:center'>"
                    f"{COPY['upload_hint']}"
                    "</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<div class='space-md'></div>", unsafe_allow_html=True)
            _, pop_col, _ = st.columns([2, 1, 2])
            with pop_col:
                with st.popover("¿Qué es un arquetipo?", use_container_width=True):
                    st.markdown(COPY["what_is_archetype"])

        return

    # Dataset loaded
    df = st.session_state["raw_df"]
    context = st.session_state.get("dataset_context", "")

    # Row A: Tipos (donut) | Contexto | Variables a usar
    cola, colb, colc = st.columns([1.5, 2, 2.5], gap="medium")

    with cola:
        with st.container(border=True):
            head_l, head_r = st.columns([3, 2])
            with head_l:
                st.markdown('<div class="panel-eyebrow">Tipos de variables</div>', unsafe_allow_html=True)
            with head_r:
                with st.popover("Cambiar archivo", use_container_width=True):
                    if advanced:
                        source = st.radio(
                            "Fuente",
                            ["Subir archivo", "Conexión SQL"],
                            horizontal=True,
                            label_visibility="collapsed",
                        )
                        if source == "Subir archivo":
                            _render_upload_block(compact=True)
                        else:
                            _render_sql_block()
                    else:
                        _render_upload_block(compact=True)
            render_type_donut(df)

    with colb:
        with st.container(border=True):
            st.markdown('<div class="panel-eyebrow">Contexto</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="panel-title">¿Qué representa este dataset?</div>',
                unsafe_allow_html=True,
            )
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
        with st.container(border=True):
            st.markdown('<div class="panel-eyebrow">Pregunta sobre tus datos</div>', unsafe_allow_html=True)
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            cat_cols = [c for c in df.columns if c not in numeric_cols]
            suggestions = ["¿Hay valores faltantes?"]
            if cat_cols:
                suggestions.append(f"¿Cuántos hay por {cat_cols[0]}?")
            if numeric_cols and cat_cols:
                suggestions.append(f"{numeric_cols[0]} promedio por {cat_cols[0]}")
            elif numeric_cols:
                suggestions.append(f"Distribución de {numeric_cols[0]}")
            render_data_chat(df, context=context, mode="raw", key="step1", suggestions=suggestions)

    if advanced:
        st.markdown("<div class='space-md'></div>", unsafe_allow_html=True)
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
