import re

from langchain_openai import ChatOpenAI

from src.config.settings import settings


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
    )


def get_llm_json() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model_kwargs={"response_format": {"type": "json_object"}},
    )


def get_narrative_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_narrative_model,
        temperature=settings.llm_temperature,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model_kwargs={"response_format": {"type": "json_object"}},
    )
