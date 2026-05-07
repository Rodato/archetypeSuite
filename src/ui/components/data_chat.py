"""Conversational data chat component (used in step 1 and step 3)."""
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from src.llm.data_qa import DataQAResult, answer_data_question

PLOTLY_TEMPLATE = "plotly_white"
BRAND_COLOR = "#4F46E5"


def _render_chart(chart: Dict[str, Any]) -> None:
    chart_type = chart.get("type") or "table"
    data: pd.DataFrame = chart.get("data")
    if data is None or chart_type in ("table", "none"):
        return

    x = chart.get("x")
    y = chart.get("y")
    color = chart.get("color")

    fig = None
    try:
        if chart_type == "bar":
            fig = px.bar(data, x=x, y=y, color=color)
        elif chart_type == "pie":
            fig = px.pie(data, names=x, values=y)
        elif chart_type == "histogram":
            fig = px.histogram(data, x=x, color=color)
        elif chart_type == "box":
            fig = px.box(data, x=x, y=y, color=color)
        elif chart_type == "scatter":
            fig = px.scatter(data, x=x, y=y, color=color)
    except Exception:
        return

    if fig is None:
        return
    try:
        fig.update_layout(
            template=PLOTLY_TEMPLATE,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", size=12, color="#475569"),
            height=320,
            margin=dict(l=20, r=10, t=30, b=20),
        )
        if chart_type in ("bar", "histogram"):
            fig.update_traces(marker_color=BRAND_COLOR)
    except Exception:
        pass
    st.plotly_chart(fig, use_container_width=True)


def _render_message(role: str, payload: Dict[str, Any]) -> None:
    with st.chat_message(role):
        if "text" in payload:
            st.markdown(payload["text"])
        if "narrative" in payload and payload["narrative"]:
            st.markdown(payload["narrative"])
        chart = payload.get("chart")
        if chart:
            _render_chart(chart)
        table = payload.get("table")
        if table is not None and not table.empty:
            st.dataframe(table, use_container_width=True, hide_index=True)
        if payload.get("error"):
            st.caption(f"⚠️ {payload['error']}")


def render_data_chat(
    df: pd.DataFrame,
    *,
    context: str = "",
    mode: str = "raw",
    key: str = "default",
    suggestions: Optional[List[str]] = None,
) -> None:
    history_key = f"chat_{key}_history"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    history: List[Dict[str, Any]] = st.session_state[history_key]

    # Render history
    for entry in history[-8:]:
        _render_message(entry["role"], entry["payload"])

    # Suggestion chips (only when chat is empty)
    if suggestions and not history:
        st.caption("Prueba con:")
        for i, q in enumerate(suggestions[:3]):
            if st.button(q, key=f"{history_key}_suggest_{i}", use_container_width=True):
                _process_question(df, q, context=context, mode=mode, history_key=history_key)
                st.rerun()

    # Input form
    with st.form(key=f"{history_key}_form", clear_on_submit=True):
        input_col, btn_col = st.columns([5, 1])
        with input_col:
            user_input = st.text_input(
                "pregunta",
                placeholder="Escribe tu pregunta aquí…",
                label_visibility="collapsed",
            )
        with btn_col:
            submitted = st.form_submit_button("Enviar", use_container_width=True)

    if submitted and user_input.strip():
        _process_question(df, user_input.strip(), context=context, mode=mode, history_key=history_key)
        st.rerun()

    if history:
        if st.button("Limpiar", key=f"{history_key}_clear", type="secondary"):
            st.session_state[history_key] = []
            st.rerun()


def _process_question(
    df: pd.DataFrame,
    question: str,
    *,
    context: str,
    mode: str,
    history_key: str,
) -> None:
    history: List[Dict[str, Any]] = st.session_state[history_key]
    history.append({"role": "user", "payload": {"text": question}})

    with st.spinner("Pensando…"):
        result: DataQAResult = answer_data_question(df, question, context=context, mode=mode)

    history.append({
        "role": "assistant",
        "payload": {
            "narrative": result.narrative,
            "table": result.table,
            "chart": result.chart,
            "error": result.error,
        },
    })
