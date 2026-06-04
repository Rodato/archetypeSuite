from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    openrouter_api_key: str = ""
    llm_model: str = "anthropic/claude-sonnet-4.5"
    llm_narrative_model: str = "x-ai/grok-4.1-fast"
    llm_temperature: float = 0.0
    llm_request_timeout: int = 60
    llm_fast_request_timeout: int = 30

    # Pipeline
    max_refinement_iterations: int = 2
    random_seed: int = 42

    # Clustering
    fixed_algorithm: str = "KMeans"
    kmeans_n_init: int = 10
    k_optimizer_min: int = 2
    k_optimizer_max: int = 10

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
