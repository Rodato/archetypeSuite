"""Microcopy centralizada — tone of voice consistente.

Convenciones:
- Imperativo de acción ("Sube tu archivo", "Ejecutar análisis"), no infinitivo decorativo.
- Sentence case en labels y CTAs (no Title Case ni MAYÚSCULAS).
- Estados de progreso terminan en "…" cuando son in-progress.
- Sin emojis salvo en feedback de éxito/celebración (✨), donde refuerzan tono.
"""

COPY = {
    # Branding
    "product_name": "Archetype Suite",
    "product_tagline": "Trae tus datos. Descubrimos los patrones de comportamiento.",

    # Onboarding (paso 1)
    "upload_label": "Sube tu archivo",
    "upload_hint": "Acepta CSV y Excel. Hasta 10 MB.",
    "demo_button": "Probar con datos de ejemplo",
    "demo_hint": "Carga un dataset de 50 clientes en 1 click.",

    # Acciones del wizard
    "analyze_button": "Ejecutar análisis",
    "analyze_again": "Volver a analizar",
    "go_to_data": "Ir a Datos",
    "go_to_analyze": "Ir a Analizar",
    "go_to_results": "Ver resultados",
    "retry": "Reintentar",

    # Estados de progreso
    "analyzing": "Analizando…",
    "loading": "Cargando…",
    "thinking": "Pensando…",
    "validating_file": "Validando archivo…",

    # Éxito
    "analysis_done": "✨ Tus arquetipos están listos",
    "file_loaded": "Archivo cargado",

    # Errores genéricos (los específicos van en datos.py:ERROR_MAP)
    "error_generic": "Algo no salió bien.",
    "error_load_file": "No pudimos leer el archivo.",
    "error_pipeline_interrupted": "El análisis se interrumpió. Reinténtalo o ajusta tus datos.",
    "error_no_columns_selected": "No seleccionaste ninguna columna. Vuelve al paso Datos.",
    "error_dataset_too_small": "El dataset es muy pequeño para encontrar patrones estables.",
    "error_no_api_key": "Falta configurar la API key de OpenRouter en tu .env.",

    # Tooltips de educación
    "what_is_archetype": (
        "Un **arquetipo** es un grupo de personas (o filas) que comparten "
        "patrones de comportamiento similares en tus datos. No es solo demografía "
        "— combina cómo se comportan, qué priorizan, y en qué se diferencian."
    ),
}
