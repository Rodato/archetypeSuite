"""Loader for the Plural behavioral methodology (knowledge_database/methodology_v1.md).

The document is injected as a system block into the interpret/refinement prompts so the
archetype narratives speak in Plural's key (behavioral patterns, not marketing personas).
Loaded once per process (lru_cache) and fail-soft: if the file is missing in a deploy that
didn't ship knowledge_database/, the pipeline still runs with a short fallback.
"""
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

METHODOLOGY_PATH = (
    Path(__file__).resolve().parents[2] / "knowledge_database" / "methodology_v1.md"
)

_FALLBACK = (
    "[Marco metodológico no disponible en este entorno. Aun así, respeta estas reglas: "
    "el arquetipo es una HIPÓTESIS COMPORTAMENTAL, no un retrato de persona. Habla de "
    "patrones ('en este grupo aparece…'), no de identidades. Usa lenguaje hipotético. "
    "Nunca nombres moralizantes ni de marketing. Aplica COM-B (capacidad/oportunidad/"
    "motivación) al hipotetizar barreras y habilitadores. Sé honesto con la cautela.]"
)


@lru_cache(maxsize=1)
def load_methodology() -> str:
    try:
        return METHODOLOGY_PATH.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        logger.warning("methodology_v1.md no encontrado (%s); usando fallback.", exc)
        return _FALLBACK
