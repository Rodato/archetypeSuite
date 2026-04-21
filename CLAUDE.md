# Archetype Suite

## Documentación (Obsidian)
Notas en: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Documentición codigo/archetypeSuite/`
Actualizar cuando cambien: nodos del pipeline, PipelineState, algoritmos de clustering, modelos LLM, prompts, páginas UI.
No actualizar por: bugfixes menores, ajustes de umbrales, cambios de copy.

## Estado del Proyecto (Apr 21, 2026)
- UI rediseñada para usuarios no-técnicos: wizard de 3 pasos (Datos, Analizar, Arquetipos) con toggle "Modo avanzado"
- Tema dark profesional con CSS custom (`.streamlit/config.toml` + `src/ui/styles.py`)
- Pipeline determinístico: `temperature=0`, `select_node` fijo en KMeans, refinement no cambia algoritmo
- Robustez LLM: helper `invoke_json_with_retry` con reintentos y fallbacks en los 3 nodos LLM
- Exportación: CSV (arquetipos, datos etiquetados) y Markdown (reporte) en paso 3
- Tests: 42/42 (40 unitarios + 2 e2e con mocks)

## Como ejecutar
- **Activar venv:** `source .venv/bin/activate`
- **Tests:** `python3 -m pytest tests/ -v`
- **UI Streamlit:** `streamlit run src/ui/app.py`
- **Requisito:** configurar `OPENROUTER_API_KEY` en `.env` antes de ejecutar el pipeline

## Arquitectura
- Pipeline LangGraph: ingest -> profile -> preprocess(LLM) -> optimize_k -> select (determinístico, KMeans) -> cluster -> evaluate -> interpret(LLM) -> refinement(LLM)
- preprocess, refinement usan Claude Sonnet 4.5 via OpenRouter
- interpret usa x-ai/grok-4.1-fast via OpenRouter (modelo narrativo separado)
- Loop de refinamiento limitado a 2 iteraciones max (fix off-by-one en `routing.py`)
- UI Streamlit — 3 pasos (en español): Datos, Analizar, Arquetipos. Archivos en `src/ui/views/` (no `pages/` para evitar auto-multipage de Streamlit).
- Clustering: KMeans, AgglomerativeClustering (DBSCAN y GaussianMixture eliminados). Select fija KMeans.

## Notas técnicas de ingesta y preprocesamiento
- `ingest.py`: validación mínima 2 filas (antes 10) — soporta Excels con pestañas pequeñas
- `preprocessor.py`: columnas categóricas se convierten a `str` antes de imputar — fix para Excels con tipos mixtos (int + str en la misma columna)
- `views/datos.py`: usa `pd.ExcelFile` para leer pestañas; si hay más de una muestra `selectbox`

## Estructura UI (`src/ui/`)
- `app.py` — entry point, sidebar con 3 pasos + toggle Modo avanzado + chip del dataset
- `styles.py` — CSS custom inyectado globalmente (tema dark, cards, sidebar)
- `quality.py` — helper `silhouette_to_quality()` (mapeo numérico → emoji/label/desc) y `natural_log_message()` (traduce logs del pipeline a mensajes amigables)
- `export.py` — `archetypes_to_csv`, `labels_to_csv`, `build_markdown_report`
- `views/datos.py` — paso 1 (upload + contexto + resumen natural; avanzado: SQL + estadísticas detalladas)
- `views/analizar.py` — paso 2 (botón + progreso en lenguaje natural; avanzado: logs crudos + razonamientos)
- `views/arquetipos.py` — paso 3 (card de calidad + cards + tabs Mapa/Comparar/Por variable + descargas)
- `components/` — `archetype_cards.py`, `cluster_plots.py` (render_quality_card, render_cluster_sizes, render_scatter_2d, render_radar_chart, render_box_plots; paleta de marca), `data_preview.py`, `profile_cards.py`

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
