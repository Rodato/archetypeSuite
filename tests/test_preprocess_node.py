"""preprocess_node determinista (sin LLM) + camino ordinal cableado (paso 1)."""
import numpy as np
import pandas as pd

from src.agents.nodes.preprocess_node import preprocess_node


def _state(df, ordinal_mappings=None):
    st = {"raw_data": df.to_dict(orient="list")}
    if ordinal_mappings is not None:
        st["ordinal_mappings"] = ordinal_mappings
    return st


class TestPreprocessNodeDeterminism:
    def test_same_input_same_output(self):
        # El corazón del producto: misma entrada → mismos datos procesados, sin LLM.
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "edad": rng.integers(18, 70, 60).astype(float),
            "ingreso": rng.integers(1000, 9000, 60).astype(float),
            "ciudad": rng.choice(["A", "B", "C"], 60),
        })
        out1 = preprocess_node(_state(df))
        out2 = preprocess_node(_state(df))
        assert out1["processed_data"] == out2["processed_data"]
        assert out1["preprocess_strategy"] == out2["preprocess_strategy"]


class TestPreprocessNodeOrdinal:
    def _mixed_df(self):
        return pd.DataFrame({
            "n": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "nivel": ["Bajo", "Alto", "Medio", "Bajo", "Alto", "Medio"],
        })

    def test_no_mapping_onehots_text(self):
        out = preprocess_node(_state(self._mixed_df()))
        cols = list(out["processed_data"].keys())
        assert any(c.startswith("nivel_") for c in cols)  # nominal → one-hot
        assert "nivel" not in cols

    def test_mapping_becomes_single_scaled_ordered_feature(self):
        out = preprocess_node(_state(self._mixed_df(), {"nivel": ["Bajo", "Medio", "Alto"]}))
        cols = list(out["processed_data"].keys())
        assert "nivel" in cols                                  # una sola columna numérica
        assert not any(c.startswith("nivel_") for c in cols)    # NO one-hot
        # Gradiente preservado tras escalar (monótono): Bajo < Medio < Alto.
        vals = out["processed_data"]["nivel"]
        bajo, alto, medio = vals[0], vals[1], vals[2]
        assert bajo < medio < alto
