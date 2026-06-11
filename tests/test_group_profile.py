"""Tests del perfilado de grupos a demanda (LLM mockeado — sin red)."""
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.llm.group_profile import (
    GroupProfileResult,
    _group_stats,
    caution_floor_for_group_size,
    profile_group,
)
from src.models.schemas import FilterCondition, GroupFilterSpec, GroupProfileDescription


@pytest.fixture
def df():
    rng = np.random.default_rng(7)
    n = 200
    return pd.DataFrame({
        "horas": rng.uniform(0.5, 8, n).round(1),
        "bienestar": rng.uniform(1, 10, n).round(1),
        "franja": rng.choice(["Mañana", "Noche", "Madrugada"], n),
        "pausas": rng.choice(["Nunca", "A veces", "Frecuentemente"], n),
    })


def _desc(nivel="baja"):
    return GroupProfileDescription(
        label="Patrón de prueba",
        description="En este grupo aparece un patrón de prueba.",
        comportamiento_principal="Conducta X",
        barreras=["barrera (capacidad física)"],
        nivel_cautela=nivel,
        cautela_reason="Test.",
    )


def _patch_llms(spec, desc=None):
    def fake_invoke(llm, prompt, model_cls, fallback):
        if model_cls is GroupFilterSpec:
            return spec, None
        return (desc or _desc()), None

    return (
        patch("src.llm.group_profile.invoke_json_with_retry", fake_invoke),
        patch("src.llm.group_profile.get_llm_json", lambda: None),
        patch("src.llm.group_profile.get_narrative_llm", lambda: None),
    )


class TestCautionFloor:
    def test_thresholds(self):
        assert caution_floor_for_group_size(10) == "alta"
        assert caution_floor_for_group_size(29) == "alta"
        assert caution_floor_for_group_size(30) == "media"
        assert caution_floor_for_group_size(99) == "media"
        assert caution_floor_for_group_size(100) == "baja"


class TestProfileGroup:
    def test_infeasible_group_returns_user_error(self, df):
        spec = GroupFilterSpec(feasible=False, reason="No hay una columna de ingresos en tus datos.")
        p1, p2, p3 = _patch_llms(spec)
        with p1, p2, p3:
            out = profile_group(df, "personas de altos ingresos")
        assert out.profile is None
        assert "ingresos" in out.error

    def test_empty_filters_returns_error(self, df):
        spec = GroupFilterSpec(filter_by=[], feasible=True)
        p1, p2, p3 = _patch_llms(spec)
        with p1, p2, p3:
            out = profile_group(df, "todo el mundo")
        assert out.error is not None

    def test_zero_rows_returns_error(self, df):
        spec = GroupFilterSpec(
            filter_by=[FilterCondition(column="horas", op="gt", value=999)],
        )
        p1, p2, p3 = _patch_llms(spec)
        with p1, p2, p3:
            out = profile_group(df, "más de 999 horas")
        assert out.profile is None
        assert "Ninguna fila" in out.error

    def test_happy_path_with_caution_floor(self, df):
        # Grupo chico (madrugada ~1/3 de 200 → <100): el LLM dice "baja" pero el
        # piso por tamaño debe subirla al menos a "media".
        spec = GroupFilterSpec(
            filter_by=[FilterCondition(column="franja", op="eq", value="Madrugada")],
            interpretation="Filtré franja = Madrugada",
        )
        p1, p2, p3 = _patch_llms(spec, _desc(nivel="baja"))
        with p1, p2, p3:
            out = profile_group(df, "quienes usan de madrugada")
        assert out.error is None
        assert out.n > 0
        assert 0 < out.share < 100
        assert out.profile["label"] == "Patrón de prueba"
        assert out.profile["nivel_cautela"] in ("media", "alta")
        assert "tamaño de muestra" in out.profile["cautela_reason"]
        assert out.filters[0]["column"] == "franja"

    def test_llm_caution_respected_when_above_floor(self, df):
        # Grupo grande (>=100) con LLM diciendo "alta": no se baja.
        spec = GroupFilterSpec(
            filter_by=[FilterCondition(column="horas", op="gte", value=0)],
        )
        p1, p2, p3 = _patch_llms(spec, _desc(nivel="alta"))
        with p1, p2, p3:
            out = profile_group(df, "todos con horas registradas")
        assert out.n == 200
        assert out.profile["nivel_cautela"] == "alta"


class TestGroupStats:
    def test_stats_compare_group_vs_total(self, df):
        group = df[df["franja"] == "Madrugada"]
        stats = _group_stats(df, group)
        assert "horas" in stats and "grupo" in stats and "total" in stats
        assert "franja" in stats
