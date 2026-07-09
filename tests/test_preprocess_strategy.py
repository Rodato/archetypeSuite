"""Estrategia de preprocesamiento determinista (reemplazo del LLM, paso 1)."""
import numpy as np
import pandas as pd

from src.data.preprocess_strategy import derive_preprocess_strategy


class TestDeriveStrategy:
    def test_default_choices(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"a": rng.normal(0, 1, 50)})
        s = derive_preprocess_strategy(df)
        assert s["encoding"] == "onehot"
        assert s["imputation"] == "median"
        assert s["scaling"] == "standard"  # sin auto-switch a robust (colas = señal en este dominio)
        assert s["drop_columns"] == []
        assert s["dimensionality_reduction"] is None
        assert s["ordinal_mappings"] == {}

    def test_scaling_is_standard_even_with_outliers(self):
        # Colas pesadas de conteos comportamentales NO deben forzar robust.
        heavy_tail = pd.DataFrame({"posts": [0, 1, 1, 2, 1, 0, 2, 1, 300, 250]})
        assert derive_preprocess_strategy(heavy_tail)["scaling"] == "standard"

    def test_ordinal_mappings_passed_through(self):
        df = pd.DataFrame({"nivel": ["Bajo", "Alto", "Medio"]})
        s = derive_preprocess_strategy(df, {"nivel": ["Bajo", "Medio", "Alto"]})
        assert s["ordinal_mappings"] == {"nivel": ["Bajo", "Medio", "Alto"]}

    def test_malformed_ordinal_mappings_ignored(self):
        # No-dict → vacío, sin abortar (evita crashear antes del fallback en el nodo).
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        assert derive_preprocess_strategy(df, ["bad"])["ordinal_mappings"] == {}
        assert derive_preprocess_strategy(df, "nope")["ordinal_mappings"] == {}

    def test_ordinal_mapping_for_missing_column_is_dropped(self):
        # Key de una columna que no está en el df → se descarta (no arrastrar keys muertas).
        df = pd.DataFrame({"existe": ["A", "B"]})
        s = derive_preprocess_strategy(df, {"existe": ["A", "B"], "no_existe": ["X", "Y"]})
        assert s["ordinal_mappings"] == {"existe": ["A", "B"]}
