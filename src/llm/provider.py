import json
import re
import time
from typing import Callable, Optional, Tuple, Type, TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from src.config.settings import settings

T = TypeVar("T", bound=BaseModel)

LLM_REQUEST_TIMEOUT = 60
RETRY_BACKOFF_SECONDS = (1, 3)


def extract_json(content: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
    if match:
        return match.group(1).strip()
    return content.strip()


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        request_timeout=LLM_REQUEST_TIMEOUT,
    )


def get_llm_json() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model_kwargs={"response_format": {"type": "json_object"}},
        request_timeout=LLM_REQUEST_TIMEOUT,
    )


def get_fast_text_llm() -> ChatOpenAI:
    """Plain text (no JSON) using the fast narrative model. For short conversational answers."""
    return ChatOpenAI(
        model=settings.llm_narrative_model,
        temperature=settings.llm_temperature,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        request_timeout=30,
    )


def get_narrative_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_narrative_model,
        temperature=settings.llm_temperature,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model_kwargs={"response_format": {"type": "json_object"}},
        request_timeout=LLM_REQUEST_TIMEOUT,
    )


def invoke_json_with_retry(
    llm: ChatOpenAI,
    prompt: str,
    schema: Type[T],
    fallback_factory: Callable[[], T],
) -> Tuple[T, Optional[str]]:
    """Llama al LLM, parsea JSON y valida con el schema. Reintenta ante fallos.

    Returns: (instancia del schema, mensaje de error si se usó fallback o None).
    """
    last_error: Optional[str] = None
    for attempt in range(len(RETRY_BACKOFF_SECONDS) + 1):
        try:
            response = llm.invoke(prompt)
            data = json.loads(extract_json(response.content))
            return schema(**data), None
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = f"{type(e).__name__}: {e}"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"

        if attempt < len(RETRY_BACKOFF_SECONDS):
            time.sleep(RETRY_BACKOFF_SECONDS[attempt])

    return fallback_factory(), last_error
