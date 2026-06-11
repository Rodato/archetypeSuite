"""Tests del chat agéntico (LLM mockeado — sin red)."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from src.config.settings import settings
from src.llm.chat_agent import answer_chat, run_agent
from src.llm.chat_tools import execute_tool
from src.llm.data_qa import DataQAResult


@pytest.fixture
def df():
    rng = np.random.default_rng(3)
    n = 60
    return pd.DataFrame({
        "edad": rng.integers(18, 70, n),
        "horas": rng.uniform(0.5, 8, n).round(1),
        "region": rng.choice(["Norte", "Sur"], n),
    })


class FakeLLM:
    """Devuelve respuestas guionadas; bind_tools devuelve el mismo objeto."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.calls += 1
        return self._responses.pop(0)


def _tool_call_msg(name, args, call_id="t1"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


class TestExecuteTool:
    def test_consultar_datos_runs_deterministic_executor(self, df):
        out = execute_tool(
            "consultar_datos",
            {"operation": "groupby_count", "groupby": ["region"], "chart_type": "bar", "narrative": "x"},
            df,
        )
        assert out.error is None
        assert out.table is not None and "conteo" in out.table.columns
        assert out.chart["type"] == "bar"

    def test_invalid_args_return_error_text_not_exception(self, df):
        out = execute_tool("consultar_datos", {"operation": "no_existe", "narrative": "x"}, df)
        assert out.error is not None
        assert "Error" in out.text

    def test_ver_arquetipos_without_archetypes(self, df):
        out = execute_tool("ver_arquetipos", {}, df, archetypes=None)
        assert "No hay arquetipos" in out.text

    def test_comparar_grupos_side_by_side(self, df):
        out = execute_tool("comparar_grupos", {
            "label_a": "Norte", "filtros_a": [{"column": "region", "op": "eq", "value": "Norte"}],
            "label_b": "Sur", "filtros_b": [{"column": "region", "op": "eq", "value": "Sur"}],
        }, df)
        assert out.error is None
        assert list(out.table.columns) == ["métrica", "Norte", "Sur"]
        assert out.table.iloc[0]["métrica"] == "filas"
        assert out.table.iloc[0]["Norte"] + out.table.iloc[0]["Sur"] == len(df)

    def test_comparar_grupos_empty_group(self, df):
        out = execute_tool("comparar_grupos", {
            "label_a": "Imposible", "filtros_a": [{"column": "edad", "op": "gt", "value": 999}],
            "label_b": "Sur", "filtros_b": [{"column": "region", "op": "eq", "value": "Sur"}],
        }, df)
        assert out.error == "grupo vacío"
        assert "Imposible" in out.text


class TestRunAgent:
    def test_tool_then_answer(self, df):
        fake = FakeLLM([
            _tool_call_msg("consultar_datos", {
                "operation": "groupby_count", "groupby": ["region"], "chart_type": "bar", "narrative": "x",
            }),
            AIMessage(content="El Norte tiene más filas que el Sur."),
        ])
        with patch("src.llm.chat_agent.get_agent_llm", lambda: fake):
            out = run_agent(df, "¿qué región tiene más filas?")
        assert "Norte" in out.narrative
        assert out.table is not None  # la tabla de la última consulta acompaña la respuesta
        assert out.chart["type"] == "bar"
        assert len(out.trace) == 1 and out.trace[0]["tool"] == "consultar_datos"

    def test_budget_cap_forces_final_answer(self, df):
        # El modelo siempre pide tools: al agotar el presupuesto se fuerza el cierre.
        loop_msg = _tool_call_msg("ver_esquema", {})
        fake = FakeLLM([loop_msg] * settings.agent_max_tool_calls + [AIMessage(content="Cierre forzado con lo que hay.")])
        with patch("src.llm.chat_agent.get_agent_llm", lambda: fake):
            out = run_agent(df, "pregunta imposible")
        assert out.narrative == "Cierre forzado con lo que hay."
        assert len(out.trace) == settings.agent_max_tool_calls

    def test_tool_error_does_not_break_loop(self, df):
        fake = FakeLLM([
            _tool_call_msg("consultar_datos", {"operation": "no_existe", "narrative": "x"}),
            AIMessage(content="No pude calcular eso, pero puedo ver la distribución por región."),
        ])
        with patch("src.llm.chat_agent.get_agent_llm", lambda: fake):
            out = run_agent(df, "algo raro")
        assert out.trace[0]["ok"] is False
        assert "región" in out.narrative


class TestAnswerChatDispatch:
    def test_flag_off_uses_one_shot(self, df, monkeypatch):
        monkeypatch.setattr(settings, "agentic_chat", False)
        sentinel = DataQAResult(narrative="one-shot", operation="count")
        with patch("src.llm.chat_agent.answer_data_question", lambda *a, **kw: sentinel) as _:
            out = answer_chat(df, "¿cuántas filas?")
        assert out.narrative == "one-shot"

    def test_agent_failure_falls_back_to_one_shot(self, df, monkeypatch):
        monkeypatch.setattr(settings, "agentic_chat", True)

        def boom(*a, **kw):
            raise RuntimeError("agente roto")

        sentinel = DataQAResult(narrative="fallback one-shot", operation="count")
        with patch("src.llm.chat_agent.run_agent", boom), \
             patch("src.llm.chat_agent.answer_data_question", lambda *a, **kw: sentinel):
            out = answer_chat(df, "¿cuántas filas?")
        assert out.narrative == "fallback one-shot"


class TestStreamAgent:
    def test_emits_tool_events_then_result(self, df):
        from src.llm.chat_agent import stream_agent

        fake = FakeLLM([
            _tool_call_msg("ver_esquema", {}),
            AIMessage(content="Respuesta final."),
        ])
        with patch("src.llm.chat_agent.get_agent_llm", lambda: fake):
            events = list(stream_agent(df, "¿qué columnas hay?"))
        assert [e["type"] for e in events] == ["tool", "result"]
        assert events[0]["tool"] == "ver_esquema" and events[0]["ok"] is True
        assert events[1]["result"].narrative == "Respuesta final."

    def test_stream_chat_flag_off_only_result(self, df, monkeypatch):
        from src.llm.chat_agent import stream_chat

        monkeypatch.setattr(settings, "agentic_chat", False)
        sentinel = DataQAResult(narrative="one-shot", operation="count")
        with patch("src.llm.chat_agent.answer_data_question", lambda *a, **kw: sentinel):
            events = list(stream_chat(df, "¿cuántas filas?"))
        assert len(events) == 1 and events[0]["type"] == "result"
        assert events[0]["result"].narrative == "one-shot"

    def test_stream_chat_falls_back_after_partial_failure(self, df, monkeypatch):
        from src.llm import chat_agent

        monkeypatch.setattr(settings, "agentic_chat", True)

        def broken_stream(*a, **kw):
            yield {"type": "tool", "tool": "ver_esquema", "args": {}, "ok": True, "summary": "..."}
            raise RuntimeError("se cayó a mitad")

        sentinel = DataQAResult(narrative="fallback", operation="count")
        with patch.object(chat_agent, "stream_agent", broken_stream), \
             patch.object(chat_agent, "answer_data_question", lambda *a, **kw: sentinel):
            events = list(chat_agent.stream_chat(df, "pregunta"))
        # 1 tool emitido + el resultado del fallback
        assert [e["type"] for e in events] == ["tool", "result"]
        assert events[1]["result"].narrative == "fallback"
