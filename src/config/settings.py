from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openrouter_api_key: str = ""
    llm_model: str = "anthropic/claude-sonnet-4.5"
    llm_narrative_model: str = "x-ai/grok-4.1-fast"
    llm_temperature: float = 0.2
    max_refinement_iterations: int = 2

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
