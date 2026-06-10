import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _dummy_api_key(monkeypatch):
    """Los tests mockean los LLM, pero las factories validan que exista una API key.

    En CI (sin .env) esto inyecta una key dummy para que la suite corra sin secretos.
    Los tests que verifican el guard de key faltante la vacían explícitamente.
    """
    from src.config.settings import settings

    if not settings.openrouter_api_key:
        monkeypatch.setattr(settings, "openrouter_api_key", "test-key")


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "age": np.random.randint(18, 70, n),
        "income": np.random.randint(20000, 120000, n),
        "spending_score": np.random.randint(1, 100, n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
        "segment": np.random.choice(["Basic", "Premium", "VIP"], n),
    })


@pytest.fixture
def numeric_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n) * 2 + 5,
        "feature_3": np.random.randn(n) * 0.5 - 1,
    })
