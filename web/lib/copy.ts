// Spanish microcopy — ported from src/ui/copy.py + a few SaaS-shell strings.
// Conventions: sentence case, in-progress ends in "…", emojis only on celebration.

export const COPY = {
  product_name: "Archetype Suite",
  product_tagline: "Trae tus datos. Descubrimos los patrones de comportamiento.",

  upload_label: "Sube tu archivo",
  upload_hint: "Acepta CSV y Excel. Hasta 10 MB.",
  upload_cta: "Arrastra un CSV o Excel, o haz clic para elegir",
  use_sample: "Probar con datos de ejemplo",

  analyze_button: "Ejecutar análisis",
  analyze_again: "Volver a analizar",
  go_to_data: "Ir a Datos",
  go_to_analyze: "Ir a Analizar",
  go_to_results: "Ver resultados",
  retry: "Reintentar",

  analyzing: "Analizando…",
  loading: "Cargando…",
  thinking: "Pensando…",
  validating_file: "Validando archivo…",

  analysis_done: "✨ Tus arquetipos están listos",
  file_loaded: "Archivo cargado",

  error_generic: "Algo no salió bien.",
  error_load_file: "No pudimos leer el archivo.",
  error_pipeline_interrupted: "El análisis se interrumpió. Reinténtalo o ajusta tus datos.",
  error_no_columns_selected: "No seleccionaste ninguna columna. Vuelve al paso Datos.",
  error_dataset_too_small: "El dataset es muy pequeño para encontrar patrones estables.",
  error_no_api_key: "Falta configurar la API key de OpenRouter en el backend.",

  what_is_archetype:
    "Un **arquetipo** es un grupo de personas (o filas) que comparten patrones de comportamiento similares en tus datos. No es solo demografía — combina cómo se comportan, qué priorizan, y en qué se diferencian.",

  // SaaS shell
  nav_new: "Nuevo análisis",
  nav_history: "Mis análisis",
  empty_history_title: "Aún no tienes análisis",
  empty_history_desc: "Sube un dataset y descubre los arquetipos que esconde.",
} as const;

export const STEP_NAMES = ["Datos", "Analizar", "Arquetipos"] as const;
