from typing import Dict, Optional


# Nivel de cautela interpretativa atado a la métrica (marco metodológico, sección 9).
CAUTION_ORDER = {"baja": 0, "media": 1, "alta": 2}

CAUTION_META = {
    "baja": {"label": "Cautela baja", "color": "green", "description": "Lectura sólida: métricas buenas y grupos bien separados."},
    "media": {"label": "Cautela media", "color": "orange", "description": "Lectura razonable, pero con vacíos: trátala como hipótesis."},
    "alta": {"label": "Cautela alta", "color": "red", "description": "Lectura exploratoria: métricas débiles o datos escasos."},
}


def caution_from_silhouette(score: Optional[float]) -> str:
    """Piso de cautela derivado del silhouette. None o <0.25 → 'alta'; <0.5 → 'media'; resto 'baja'."""
    if score is None or score < 0.25:
        return "alta"
    if score < 0.5:
        return "media"
    return "baja"


def silhouette_to_quality(score: Optional[float]) -> Dict[str, str]:
    if score is None:
        return {
            "grade": "—",
            "label": "Sin calcular",
            "description": "No se pudo calcular la calidad del análisis.",
            "color": "gray",
        }
    if score >= 0.5:
        return {
            "grade": "A",
            "label": "Excelente",
            "description": "Los arquetipos están muy bien diferenciados entre sí.",
            "color": "green",
        }
    if score >= 0.3:
        return {
            "grade": "B",
            "label": "Buena",
            "description": "Los arquetipos son claramente distinguibles.",
            "color": "green",
        }
    if score >= 0.15:
        return {
            "grade": "C",
            "label": "Aceptable",
            "description": "Hay cierto solapamiento entre arquetipos, pero la segmentación es útil.",
            "color": "orange",
        }
    return {
        "grade": "D",
        "label": "Baja",
        "description": "Los arquetipos se solapan mucho. Considera más datos o variables más diferenciadoras.",
        "color": "red",
    }


NODE_FRIENDLY_MESSAGES = {
    "ingest": "Cargando tus datos…",
    "profile": "Entendiendo tus datos…",
    "column_selection": "Aplicando selección de variables…",
    "preprocess": "Preparando datos…",
    "optimize_k": "Buscando el mejor número de grupos…",
    "select": "Eligiendo método de análisis…",
    "cluster": "Encontrando patrones…",
    "evaluate": "Validando resultados…",
    "interpret": "Describiendo arquetipos…",
    "refinement": "Ajustando…",
}

# Pasos visibles en la UI durante el análisis (orden = orden de ejecución del grafo).
# `key` debe coincidir con el prefijo `[clave]` que cada nodo escribe en log_messages.
PIPELINE_UI_STEPS = [
    ("ingest", "Cargar datos"),
    ("profile", "Entender el dataset"),
    ("column_selection", "Filtrar columnas"),
    ("preprocess", "Limpiar y normalizar"),
    ("optimize_k", "Buscar k óptimo"),
    ("cluster", "Encontrar patrones"),
    ("interpret", "Generar arquetipos"),
    ("refinement", "Refinar narrativa"),
]


def natural_log_message(raw_log: str) -> Optional[str]:
    """Convierte un log técnico ([nodo] mensaje) a lenguaje natural. None si se debe ocultar."""
    if not raw_log.startswith("["):
        return None
    end = raw_log.find("]")
    if end < 0:
        return None
    node_name = raw_log[1:end]
    return NODE_FRIENDLY_MESSAGES.get(node_name)


def nodes_with_logs(logs: list) -> set:
    """Devuelve el set de nodos que ya escribieron al menos un log."""
    seen = set()
    for raw in logs:
        if raw.startswith("[") and "]" in raw:
            seen.add(raw[1:raw.find("]")])
    return seen
