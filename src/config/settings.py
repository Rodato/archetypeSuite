from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM
    openrouter_api_key: str = ""
    llm_model: str = "anthropic/claude-sonnet-4.5"
    llm_narrative_model: str = "x-ai/grok-4.3"
    llm_temperature: float = 0.0
    llm_request_timeout: int = 60
    llm_fast_request_timeout: int = 30

    # Pipeline
    max_refinement_iterations: int = 2
    # Gate determinista de refinamiento: silhouette bajo el umbral → 1 reintento exhaustivo.
    refinement_silhouette_threshold: float = 0.25
    refinement_n_init: int = 30
    random_seed: int = 42

    # Clustering
    fixed_algorithm: str = "KMeans"
    kmeans_n_init: int = 10
    k_optimizer_min: int = 2
    k_optimizer_max: int = 10
    # Regla de dos regímenes: si la curva de silhouette es plana (max-min < rango),
    # los datos no distinguen ningún k → preferir "pocos y trabajables" (k <= flat_max_k).
    k_flat_curve_range: float = 0.03
    k_flat_max_k: int = 4

    # Chat agéntico (capa narrativa): loop de tools deterministas con presupuesto acotado.
    agentic_chat: bool = True
    agent_max_tool_calls: int = 5

    # Preprocessing guards
    enable_pca: bool = False           # PCA off by default — avoids the 1-D collapse that inflates silhouette
    max_onehot_levels: int = 12        # cap one-hot categories per column (rare grouped into "infrequent")
    max_categorical_cardinality: int = 25  # drop categoricals with more unique values than this (explode one-hot)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
