import numpy as np
import pandas as pd
import pytest


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
