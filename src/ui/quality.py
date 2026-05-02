from typing import Dict, Optional


def silhouette_to_quality(score: Optional[float]) -> Dict[str, str]:
    if score is None:
        return {
            "emoji": "❓",
            "label": "Sin calcular",
            "description": "No se pudo calcular la calidad del análisis.",
            "color": "gray",
        }

    if score >= 0.5:
        return {
            "emoji": "🌟",
            "label": "Excelente",
            "description": "Los arquetipos están muy bien diferenciados entre sí.",
            "color": "green",
        }
    if score >= 0.3:
        return {
            "emoji": "✅",
            "label": "Buena",
            "description": "Los arquetipos son claramente distinguibles.",
            "color": "green",
        }
    if score >= 0.15:
        return {
            "emoji": "⚠️",
            "label": "Aceptable",
            "description": "Hay cierto solapamiento entre arquetipos, pero la segmentación es útil.",
            "color": "orange",
        }
    return {
        "emoji": "❌",
        "label": "Baja",
        "description": "Los arquetipos se solapan mucho. Considera más datos o variables más diferenciadoras.",
        "color": "red",
    }


NODE_FRIENDLY_MESSAGES = {
    "ingest": "📥 Cargando tus datos...",
    "profile": "🔍 Entendiendo tus datos...",
    "column_selection": "🎛️ Aplicando selección de variables...",
    "preprocess": "🧹 Preparando datos...",
    "optimize_k": "📐 Buscando el mejor número de grupos...",
    "select": "🎯 Eligiendo método de análisis...",
    "cluster": "🔗 Encontrando patrones...",
    "evaluate": "✅ Validando resultados...",
    "interpret": "📝 Describiendo arquetipos...",
    "refinement": "🔄 Ajustando...",
}


def natural_log_message(raw_log: str) -> Optional[str]:
    """Convierte un log técnico ([nodo] mensaje) a lenguaje natural. None si se debe ocultar."""
    if not raw_log.startswith("["):
        return None
    end = raw_log.find("]")
    if end < 0:
        return None
    node_name = raw_log[1:end]
    return NODE_FRIENDLY_MESSAGES.get(node_name)
