"""Tests for the behavioral methodology layer: loader, caution thresholds, schema compat."""
from src.agents.nodes.interpret_node import _fallback_interpretation
from src.llm.methodology import load_methodology
from src.models.schemas import ArchetypeDescription
from src.core.quality import CAUTION_META, CAUTION_ORDER, caution_from_silhouette


class TestMethodologyLoader:
    def test_returns_non_empty_with_markers(self):
        text = load_methodology()
        assert len(text) > 500
        assert "hipótesis comportamental" in text
        assert "Reglas duras para el LLM" in text

    def test_is_cached(self):
        assert load_methodology() is load_methodology()  # lru_cache → same object


class TestCautionFromSilhouette:
    def test_thresholds(self):
        assert caution_from_silhouette(None) == "alta"
        assert caution_from_silhouette(0.10) == "alta"
        assert caution_from_silhouette(0.24) == "alta"
        assert caution_from_silhouette(0.25) == "media"
        assert caution_from_silhouette(0.49) == "media"
        assert caution_from_silhouette(0.50) == "baja"
        assert caution_from_silhouette(0.80) == "baja"

    def test_order_and_meta_consistent(self):
        assert CAUTION_ORDER == {"baja": 0, "media": 1, "alta": 2}
        for level in ("baja", "media", "alta"):
            assert level in CAUTION_META
            assert CAUTION_META[level]["color"] in ("green", "orange", "red")


class TestArchetypeSchema:
    def test_old_four_field_dict_validates_with_defaults(self):
        # A pre-methodology persisted run (4-field) must still validate.
        old = {
            "cluster_id": 0,
            "label": "Familias Ahorrativas",
            "description": "Descripción antigua",
            "key_characteristics": ["a", "b"],
            "differentiators": ["c"],
        }
        a = ArchetypeDescription(**old)
        assert a.nivel_cautela == "media"
        assert a.microcomportamientos == []
        assert a.barreras == []
        assert a.key_characteristics == ["a", "b"]

    def test_eight_field_dict_validates(self):
        a = ArchetypeDescription(
            cluster_id=1,
            label="Estudio interrumpido",
            description="En este grupo aparece un patrón de estudio fragmentado.",
            comportamiento_principal="Sesiones breves y discontinuas.",
            microcomportamientos=["Revisar el celular durante el estudio"],
            barreras=["Sobrecarga de plazos (oportunidad física)"],
            habilitadores=["Pares que sostienen rutinas"],
            oportunidades_accion=["Explorar ventanas cortas de foco"],
            nivel_cautela="alta",
            cautela_reason="Métricas débiles.",
        )
        assert a.nivel_cautela == "alta"
        assert len(a.barreras) == 1


class TestFallbackInterpretation:
    def test_fallback_is_high_caution_and_shaped(self):
        result = _fallback_interpretation(2)
        assert len(result.archetypes) == 2
        for a in result.archetypes:
            assert a.nivel_cautela == "alta"
            assert a.microcomportamientos == []
            assert a.cautela_reason
