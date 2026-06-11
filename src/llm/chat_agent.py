"""Chat agéntico: loop ReAct acotado sobre tools deterministas.

Arquitectura de dos capas: los números los producen las tools (executor
whitelisteado de data_qa); el agente solo decide QUÉ consultar y redacta.
Presupuesto duro de tool-calls + fallback al chat one-shot si el loop falla.

El núcleo es un GENERADOR (`stream_agent`) que emite cada tool-call al ejecutarse
— el endpoint SSE los retransmite en vivo y la UI muestra "consultando…".
"""
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.config.settings import settings
from src.llm.chat_tools import TOOL_SCHEMAS, execute_tool
from src.llm.data_qa import MODE_DESCRIPTIONS, DataQAResult, answer_data_question
from src.llm.prompts import CHAT_AGENT_SYSTEM_PROMPT
from src.llm.provider import get_agent_llm


def _history_messages(history: Optional[List[Dict[str, str]]], max_entries: int = 6) -> List[Any]:
    messages: List[Any] = []
    for entry in (history or [])[-max_entries:]:
        text = (entry.get("text") or "").strip()[:280]
        if not text:
            continue
        cls = HumanMessage if entry.get("role") == "user" else AIMessage
        messages.append(cls(content=text))
    return messages


def stream_agent(
    df: pd.DataFrame,
    question: str,
    *,
    context: str = "",
    mode: str = "raw",
    history: Optional[List[Dict[str, str]]] = None,
    archetypes: Optional[List[Dict[str, Any]]] = None,
) -> Iterator[Dict[str, Any]]:
    """Núcleo del agente. Emite eventos:
    - {"type": "tool", "tool", "args", "ok", "summary"} tras cada tool-call ejecutada
    - {"type": "result", "result": DataQAResult} al final (siempre, exactamente uno)
    """
    max_steps = settings.agent_max_tool_calls
    base_llm = get_agent_llm()
    llm = base_llm.bind_tools(TOOL_SCHEMAS)

    system = CHAT_AGENT_SYSTEM_PROMPT.format(
        mode_description=MODE_DESCRIPTIONS.get(mode, MODE_DESCRIPTIONS["raw"]),
        context=context or "No se proporcionó contexto adicional.",
        n_rows=len(df),
        columns=", ".join(str(c) for c in df.columns[:30]),
        max_steps=max_steps,
    )
    messages: List[Any] = [SystemMessage(content=system)]
    messages.extend(_history_messages(history))
    messages.append(HumanMessage(content=question))

    trace: List[Dict[str, Any]] = []
    last_table: Optional[pd.DataFrame] = None
    last_chart: Optional[Dict[str, Any]] = None
    used_calls = 0

    while used_calls < max_steps:
        response = llm.invoke(messages)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            narrative = (response.content or "").strip()
            if narrative:
                yield {"type": "result", "result": DataQAResult(
                    narrative=narrative, operation="agent",
                    table=last_table, chart=last_chart, trace=trace,
                )}
                return
            break  # respuesta vacía sin tools: salir al cierre forzado

        messages.append(response)
        for tc in tool_calls:
            used_calls += 1
            name = tc.get("name", "")
            args = tc.get("args") or {}
            execution = execute_tool(name, args, df, archetypes)
            if execution.table is not None:
                last_table, last_chart = execution.table, execution.chart
            step = {
                "tool": name,
                "args": args,
                "ok": execution.error is None,
                "summary": execution.text[:200],
            }
            trace.append(step)
            yield {"type": "tool", **step}
            messages.append(ToolMessage(content=execution.text, tool_call_id=tc.get("id") or name))

    # Presupuesto agotado (o respuesta vacía): cierre forzado sin tools.
    messages.append(HumanMessage(
        content="Responde ahora a la pregunta original con la evidencia que ya tienes, sin usar más herramientas.",
    ))
    response = base_llm.invoke(messages)
    narrative = (response.content or "").strip() or "No pude completar el análisis — intenta reformular la pregunta."
    yield {"type": "result", "result": DataQAResult(
        narrative=narrative, operation="agent",
        table=last_table, chart=last_chart, trace=trace,
    )}


def run_agent(
    df: pd.DataFrame,
    question: str,
    *,
    context: str = "",
    mode: str = "raw",
    history: Optional[List[Dict[str, str]]] = None,
    archetypes: Optional[List[Dict[str, Any]]] = None,
) -> DataQAResult:
    """Versión síncrona: consume el stream y devuelve solo el resultado final."""
    final: Optional[DataQAResult] = None
    for event in stream_agent(
        df, question, context=context, mode=mode, history=history, archetypes=archetypes,
    ):
        if event["type"] == "result":
            final = event["result"]
    assert final is not None  # stream_agent siempre termina con un result
    return final


def stream_chat(
    df: pd.DataFrame,
    question: str,
    *,
    context: str = "",
    mode: str = "raw",
    history: Optional[List[Dict[str, str]]] = None,
    archetypes: Optional[List[Dict[str, Any]]] = None,
) -> Iterator[Dict[str, Any]]:
    """Punto de entrada streaming: agente si está activado, con fallback fail-soft
    al one-shot (que no emite eventos de tools, solo el resultado)."""
    if not settings.agentic_chat:
        yield {"type": "result", "result": answer_data_question(
            df, question, context=context, mode=mode, history=history,
        )}
        return
    try:
        yield from stream_agent(
            df, question, context=context, mode=mode, history=history, archetypes=archetypes,
        )
    except Exception:  # noqa: BLE001 — el chat nunca debe romperse por el agente
        yield {"type": "result", "result": answer_data_question(
            df, question, context=context, mode=mode, history=history,
        )}


def answer_chat(
    df: pd.DataFrame,
    question: str,
    *,
    context: str = "",
    mode: str = "raw",
    history: Optional[List[Dict[str, str]]] = None,
    archetypes: Optional[List[Dict[str, Any]]] = None,
) -> DataQAResult:
    """Punto de entrada no-streaming (endpoints /chat clásicos)."""
    if not settings.agentic_chat:
        return answer_data_question(df, question, context=context, mode=mode, history=history)
    try:
        return run_agent(
            df, question, context=context, mode=mode, history=history, archetypes=archetypes,
        )
    except Exception:  # noqa: BLE001
        return answer_data_question(df, question, context=context, mode=mode, history=history)
