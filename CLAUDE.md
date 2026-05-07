# Archetype Suite

## Documentación (Obsidian)
Notas en: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Documentición codigo/archetypeSuite/`
Actualizar cuando cambien: nodos del pipeline, PipelineState, algoritmos de clustering, modelos LLM, prompts, páginas UI.
No actualizar por: bugfixes menores, ajustes de umbrales, cambios de copy.

## Estado del Proyecto (May 5, 2026)
- Pipeline de **10 nodos** (insertado `column_selection_node` entre `profile` y `preprocess`, May 2026)
- **Selección inteligente de variables** (May 2026): filtros estáticos deterministas (IDs, free-text, fechas, alta missing, varianza-cero) + sugerencia LLM contextual + checkboxes en UI con badges de importancia
- **Transparencia del por qué** (May 2026): curva de silueta vs k visible (expander "¿Por qué N arquetipos?"), prevalencia % en cards, popover "ℹ️ ¿Qué es esto?", botón "Probar con datos de ejemplo"
- **Chat conversacional** (May 2026) en paso 1 (datos crudos, modo `raw`) y paso 3 (sobre arquetipos, modo `archetypes`). LLM elige operación pandas estructurada (`DataQuery`), Python ejecuta determinísticamente — nunca se ejecuta código LLM-generado
  - `groupby_agg` soporta **múltiples columnas objetivo** — "muéstrame las diferencias entre arquetipos" agrega todas las numéricas relevantes en una sola tabla
  - `DataQuery.bins` permite **rangos personalizados** vía `pd.cut` (ej. "edad agrupada 16-19 y 20-25") aplicados antes del groupby
  - `DataQuery.columns` tolera `null` del LLM (validator coerce → `[]`) — fix `ValidationError` que rompía preguntas complejas
- UI wizard 3 pasos (Datos, Analizar, Arquetipos) con toggle "Modo avanzado"
- Tema dark profesional con CSS custom (`.streamlit/config.toml` + `src/ui/styles.py`)
- Pipeline determinístico: `temperature=0`, `select_node` fijo en KMeans, refinement no cambia algoritmo
- Robustez LLM: helper `invoke_json_with_retry` con reintentos y fallbacks en los **4 nodos LLM** + chat Q&A
- Exportación: CSV (arquetipos, datos etiquetados) y Markdown (reporte) en paso 3
- Tests: **74/74** (42 originales + 17 column_filter + 4 column_selection + 11 data_qa)

**Pendiente (acordado con usuario, May 2026):**
- Memoria entre corridas (persistencia + comparación) — diferido, decidir profundidad después
- Marco Metodológico — equipo metodológico lo construye en paralelo; cuando entreguen `methodology_v1.md`, inyectar en `INTERPRETATION_PROMPT` + `REFINEMENT_PROMPT` y ampliar `ArchetypeDescription` con tensión/motivadores/barreras/heurísticas/detonante/palanca

## Como ejecutar
- **Activar venv:** `source .venv/bin/activate`
- **Tests:** `python3 -m pytest tests/ -v`
- **UI Streamlit:** `streamlit run src/ui/app.py`
- **Requisito:** configurar `OPENROUTER_API_KEY` en `.env` antes de ejecutar el pipeline

## Arquitectura
- Pipeline LangGraph: ingest -> profile -> **column_selection(LLM)** -> preprocess(LLM) -> optimize_k -> select (determinístico, KMeans) -> cluster -> evaluate -> interpret(LLM) -> refinement(LLM)
- column_selection, preprocess, refinement, data_qa usan Claude Sonnet 4.5 via OpenRouter
- interpret usa x-ai/grok-4.1-fast via OpenRouter (modelo narrativo separado)
- Loop de refinamiento limitado a 2 iteraciones max (fix off-by-one en `routing.py`)
- UI Streamlit — 3 pasos (en español): Datos, Analizar, Arquetipos. Archivos en `src/ui/views/` (no `pages/` para evitar auto-multipage de Streamlit).
- Clustering: KMeans, AgglomerativeClustering (DBSCAN y GaussianMixture eliminados). Select fija KMeans.

## Notas técnicas de ingesta y preprocesamiento
- `ingest.py`: validación mínima 2 filas (antes 10) — soporta Excels con pestañas pequeñas
- `preprocessor.py`: columnas categóricas se convierten a `str` antes de imputar — fix para Excels con tipos mixtos (int + str en la misma columna)
- `views/datos.py`: usa `pd.ExcelFile` para leer pestañas; si hay más de una muestra `selectbox`
- `column_filter.py` (May 2026): filtros estáticos deterministas. **Orden importante: datetime ANTES que id**, porque las fechas tienen cardinalidad alta y serían marcadas como ID. ID heuristic distingue numéricas continuas (income con todos únicos) de IDs (enteros secuenciales con `diffs == 1` o nombre matchea regex)
- `column_selection_node.py` (May 2026): expone `suggest_columns(df, context)` como función pura para que la UI llame antes del pipeline; el nodo respeta `selected_columns` upstream del estado o, si no, usa la sugerencia LLM como selección final

## Estructura UI (`src/ui/`)
- `app.py` — entry point, sidebar con 3 pasos + toggle Modo avanzado + chip del dataset
- `styles.py` — CSS custom inyectado globalmente (tema dark, cards, sidebar, chip `.prevalence`)
- `quality.py` — helper `silhouette_to_quality()` (mapeo numérico → emoji/label/desc) y `natural_log_message()` (traduce logs del pipeline a mensajes amigables; incluye `column_selection`)
- `export.py` — `archetypes_to_csv`, `labels_to_csv`, `build_markdown_report`
- `views/datos.py` — paso 1 (upload + botón demo + contexto + resumen natural + chat Q&A + sección "Variables a usar"; avanzado: SQL + estadísticas detalladas)
- `views/analizar.py` — paso 2 (botón + progreso en lenguaje natural; pasa `selected_columns` + `static_filter_result` + `column_recommendation` al pipeline)
- `views/arquetipos.py` — paso 3 (popover "¿Qué es esto?" + card de calidad + expander curva silueta + cards con prevalencia + tabs Mapa/Comparar/Por variable/**Conversar** + descargas)
- `components/` — `archetype_cards.py` (con chip prevalencia), `cluster_plots.py` (incluye `render_silhouette_curve`), `column_selector.py` (May 2026), `data_chat.py` (May 2026), `data_preview.py`, `profile_cards.py`

## Marco Metodológico (pendiente de construir)
Objetivo: consolidar los múltiples documentos metodológicos existentes en un marco único que se inyecte en los prompts del pipeline (principalmente `interpret`, `refinement`, `select`) para que los arquetipos y narrativas reflejen la metodología propia del estudio.

**Secciones que debe tener el documento metodológico:**
1. **Fundamentos** — definición propia de "arquetipo" (qué es y qué no es), unidad de análisis.
2. **Marco teórico** — corrientes de ciencias del comportamiento en uso (COM-B, EAST, Fogg, Dual Process, SDT, etc.), glosario operativo de términos que el LLM debe usar.
3. **Schema del arquetipo** — campos fijos (nombre, tensión, motivadores, barreras, heurísticas, contexto detonante, palanca de cambio), longitud y tono por campo, ejemplos buenos/malos.
4. **Narrativa** — voz, tono, qué debe explicar (porqué conductual, teoría, implicación de diseño), formato.
5. **Criterios de calidad** — checklist de arquetipo bien formado, errores típicos (ej. confundir demografía con conducta).
6. **Ejemplos canónicos** — 2-3 arquetipos reales de proyectos pasados.
7. **Alcance** — qué decisiones no toma el arquetipo.

**Plan de construcción:**
1. Inventario de docs existentes, etiquetados por sección.
2. Extracción de fragmentos relevantes con trazabilidad a fuente.
3. Detección de conflictos entre docs → decisiones explícitas.
4. Consolidación en documento único.
5. Validación contra 2-3 proyectos pasados.
6. Versionar como `methodology_v1.md` en el repo.

Documento completo en vault: `06 - archetypeSuite - Marco Metodológico.md`.

## Notas tecnicas
- Python 3.9 - se usa `typing.Dict`, `typing.List`, `typing.Optional` en vez de `dict | None` syntax
- En `src/models/state.py` NO usar `from __future__ import annotations` porque LangGraph evalua type hints en runtime
- En `src/models/schemas.py` (Pydantic) tampoco usar `from __future__ import annotations`
- Dataset de ejemplo: `sample_data/customers.csv` (50 filas, 8 columnas)
- Los modelos via OpenRouter a veces ignoran `response_format: json_object` y envuelven en markdown — usar siempre `extract_json()` al parsear respuestas LLM
