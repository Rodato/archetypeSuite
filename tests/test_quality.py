"""Regresiones del piso de cautela y el grade frente a métricas inválidas (NaN).

Bug del round adversarial: `nan < t` y `nan >= t` son ambas False, así que un NaN
se colaba a la rama de baja cautela / grade alto en vez de tratarse como métrica
ausente. El guard `score_unavailable` unifica el trato de None/NaN/no-numérico.
"""
import numpy as np

from src.core.quality import (
    caution_from_silhouette,
    score_unavailable,
    silhouette_to_quality,
)


class TestScoreUnavailable:
    def test_none_is_unavailable(self):
        assert score_unavailable(None) is True

    def test_nan_is_unavailable(self):
        assert score_unavailable(float("nan")) is True

    def test_valid_float_is_available(self):
        assert score_unavailable(0.3) is False

    def test_valid_int_is_available(self):
        assert score_unavailable(0) is False

    def test_numpy_float_is_available(self):
        # np.float32 NO es subclase de float de Python → antes se marcaba "no disponible".
        assert score_unavailable(np.float32(0.3)) is False
        assert score_unavailable(np.float64(0.3)) is False

    def test_numpy_nan_is_unavailable(self):
        assert score_unavailable(np.float32("nan")) is True
        assert score_unavailable(np.float64("nan")) is True

    def test_bool_is_unavailable(self):
        # bool hereda de int; True/False no son un silhouette válido.
        assert score_unavailable(True) is True
        assert score_unavailable(False) is True

    def test_inf_is_unavailable(self):
        # El silhouette está acotado en [-1, 1]; un inf indica un bug upstream, no una métrica.
        assert score_unavailable(float("inf")) is True
        assert score_unavailable(float("-inf")) is True

    def test_gigantic_int_does_not_crash(self):
        # El guard debe ser total: un int que no cabe en float → 'no disponible', no OverflowError.
        assert score_unavailable(10 ** 400) is True


class TestCautionFromSilhouette:
    def test_nan_floors_to_alta(self):
        # Regresión: antes devolvía 'baja' (NaN se colaba por debajo de todos los umbrales).
        assert caution_from_silhouette(float("nan")) == "alta"

    def test_none_is_alta(self):
        assert caution_from_silhouette(None) == "alta"

    def test_thresholds_unchanged(self):
        assert caution_from_silhouette(0.10) == "alta"
        assert caution_from_silhouette(0.24) == "alta"
        assert caution_from_silhouette(0.25) == "media"
        assert caution_from_silhouette(0.49) == "media"
        assert caution_from_silhouette(0.50) == "baja"


class TestSilhouetteToQuality:
    def test_nan_is_sin_calcular(self):
        # Regresión: antes devolvía grade 'D' / 'Baja' para NaN.
        q = silhouette_to_quality(float("nan"))
        assert q["grade"] == "—"
        assert q["label"] == "Sin calcular"

    def test_none_is_sin_calcular(self):
        assert silhouette_to_quality(None)["grade"] == "—"

    def test_grades_unchanged(self):
        assert silhouette_to_quality(0.55)["grade"] == "A"
        assert silhouette_to_quality(0.35)["grade"] == "B"
        assert silhouette_to_quality(0.20)["grade"] == "C"
        assert silhouette_to_quality(0.05)["grade"] == "D"
