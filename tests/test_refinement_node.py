"""Gate determinista de refinement: umbral de silhouette + un único reintento.

Cubre el bug del round adversarial (NaN clasificado como 'separación aceptable')
y los invariantes del gate: refina a lo sumo una vez y termina siempre.
"""
import numpy as np

from src.agents.nodes.refinement_node import refinement_node
from src.config.settings import settings


def _state(silhouette, count=0):
    return {"refinement_count": count, "metrics": {"silhouette_score": silhouette}}


class TestRefinementGate:
    def test_weak_separation_triggers_one_retry(self):
        out = refinement_node(_state(0.10, count=0))
        assert out["should_refine"] is True
        assert out["algorithm_params"]["n_init"] == settings.refinement_n_init

    def test_acceptable_separation_does_not_refine(self):
        out = refinement_node(_state(0.40, count=0))
        assert out["should_refine"] is False
        assert "aceptable" in out["refinement_reason"].lower()

    def test_second_pass_never_refines(self):
        # count>0 → ya se aplicó el reintento, se conserva el resultado.
        out = refinement_node(_state(0.10, count=1))
        assert out["should_refine"] is False

    def test_none_metric_does_not_refine(self):
        out = refinement_node(_state(None, count=0))
        assert out["should_refine"] is False
        assert "sin métrica" in out["refinement_reason"].lower()

    def test_nan_metric_is_treated_as_missing(self):
        # Regresión: antes daba 'Separación aceptable (silhouette nan ≥ 0.25)'.
        out = refinement_node(_state(float("nan"), count=0))
        assert out["should_refine"] is False
        assert "nan" not in out["refinement_reason"].lower()
        assert "aceptable" not in out["refinement_reason"].lower()

    def test_numpy_scalar_metric_triggers_retry(self):
        # np.float32 NO es subclase de float de Python; el gate debe tratarlo como métrica
        # válida igual que un float nativo (regresión: `isinstance(_, (int, float))` lo
        # descartaba → caía en 'Sin métrica' y no refinaba pese a estar bajo el umbral).
        out = refinement_node(_state(np.float32(0.10), count=0))
        assert out["should_refine"] is True
        assert out["algorithm_params"]["n_init"] == settings.refinement_n_init

    def test_numpy_nan_metric_is_treated_as_missing(self):
        out = refinement_node(_state(np.float64("nan"), count=0))
        assert out["should_refine"] is False
        assert "sin métrica" in out["refinement_reason"].lower()
