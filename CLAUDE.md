# Archetype Suite

## Documentación (Obsidian)
Notas en: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Documentición codigo/archetypeSuite/`
Actualizar cuando cambien: nodos del pipeline, PipelineState, algoritmos de clustering, modelos LLM, prompts, páginas UI.
No actualizar por: bugfixes menores, ajustes de umbrales, cambios de copy.

## Estado del Proyecto (May 7, 2026)

### Pipeline (sin cambios estructurales recientes)
- **10 nodos** LangGraph: ingest → profile → column_selection(LLM) → preprocess(LLM) → optimize_k → select(determinístico, KMeans) → cluster → evaluate → interpret(LLM) → refinement(LLM)
- Determinístico (`temperature=0`, `random_state=42`, KMeans fijo). Misma entrada → mismo arquetipo.
- 4 nodos LLM usan `invoke_json_with_retry` con reintentos + fallback determinista.
- Loop de refinamiento limitado a 2 iteraciones (`max_refinement_iterations` en settings).

### Round de polish UI (May 6-7, 2026) — branch `redesign/ui-wizard-pipeline-determinista`
13 commits que llevaron el proyecto a "look & feel SaaS pulido". Plan original en `~/.claude/plans/starry-munching-rose.md`.

**Limpieza foundational:**
- Eliminados como código muerto: `get_llm()`, `ALGORITHM_SELECTION_PROMPT`, `preprocessing_metadata` field del state, `error` field del state.
- Centralizadas configs hardcoded en `Settings`: `fixed_algorithm`, `kmeans_n_init`, `k_optimizer_min/max`, `llm_request_timeout`, `llm_fast_request_timeout`, `random_seed`. Antes vivían en cada nodo.
- `_ensure_api_key()` en `provider.py` valida `OPENROUTER_API_KEY` y lanza `RuntimeError` claro si falta (las 3 factory functions lo invocan).
- `optimize_k_node` con guards: `n_samples < 4` o `n_features < 1` → `ValueError` con mensaje user-friendly.
- Mensajes de `DataIngestor.validate()` traducidos a español user-facing.
- Creados `.env.example` y `README.md` mínimo.

**UI:**
- **Paso 1 single-page**: Row A (donut de tipos · contexto · variables) + Row B (preview 5 filas · chat). Eliminado "Resumen natural" duplicado.
- **Donut de tipos** (`render_type_donut` en `data_preview.py`): Plotly hole=0.62 con 4 categorías (Numéricas/Categóricas/Fechas/Texto libre — texto libre = >50% valores únicos). Total al centro.
- **Botón demo eliminado**: el usuario lo decidió. Para cargar demo, drag&drop manual de `sample_data/customers.csv`. Cambiar archivo accesible en popover `⋯` siempre visible en header del card del donut.
- **Wizard progress bar**: barra de 3 segmentos coloreados según estado + texto grande "Paso N de 3 · {nombre}". Reemplaza los pills viejos.
- **Empty state del paso 1**: hero con tagline + brand mark con gradient + CTA upload + popover educativo "¿Qué es un arquetipo?".
- **Variables a usar**: header "✨ La IA recomendó N de M columnas" + leyenda inline "Alta clave · Media aporta · Baja útil con pocas columnas" con chips de muestra + tooltips en cada badge. Lista en `st.expander(expanded=True)` para que se pueda colapsar. Container con scroll interno (`st.container(height=320)`) cuando hay 6+ recomendaciones — el card no empuja el chat. Tres expanders consistentes: Filtros automáticos previos / Recomendadas / Otras disponibles. Contador en vivo "X de N seleccionadas".
- **Chat con scroll interno**: el historial vive en `st.container(height=320)`, el input siempre visible abajo. Aplica al chat del paso 1 y al tab Conversar del paso 3.
- **Vista previa**: `MAX_PREVIEW_ROWS = 5` (antes 100) + `height` explícito en `st.dataframe` para que el contenedor mida exactamente 5 filas (sin área scrollable vacía).
- **Paso 2 (Analizar)**: checklist visible de 8 pasos del pipeline con estado per-step (pending → running con animación pulse → done con ✓). `try/except` global con mensaje humano + retry. Validación temprana de API key. Panel de éxito ✨ con animación fade-in. `LOAD_ERROR_MAP` traduce errores comunes (ParserError, UnicodeDecodeError, etc.) a mensajes user-friendly.
- **Paso 3 (Arquetipos)**: cards compactas con expander "Ver detalles" para características y diferenciadores. Descargas movidas a popover "📥 Descargar" en header. Modo avanzado agrupado en un único expander "Detalles técnicos". Microcopy del expander "¿Por qué N arquetipos?" reescrita en lenguaje humano. Sugerencias del chat con nombres reales: "Diferencias clave entre {arch_a} y {arch_b}".

**Sistema visual:**
- Favicon `◆` y brand mark con gradient en topbar.
- Sistema de tokens CSS documentado: COLOR SYSTEM, SPACING SCALE, TYPE SCALE, RADII.
- `--text-muted` subido a `#64748B` (5.5:1 contra `--surface-elevated` — antes fallaba WCAG AA con 3.5:1).
- Clases `.qh-grade--{green/orange/red/gray}`, `.space-{xs,sm,md,lg,xl}` y `.var-list` centralizadas en CSS (antes inline-styles).
- `panel--accent` quitado del card Contexto en datos.py (competía con Variables). Mantenido en analizar.py donde sí indica jerarquía ("esto está pasando ahora").
- Custom scrollbar 6px color `--border-strong` para los containers con scroll interno.

**Microcopy centralizada en `src/ui/copy.py`** con tone of voice consistente (sentence case, ellipsis "…" en in-progress, emojis solo en celebración).

**Tests: 80/80** (74 originales + 6 nuevos en `tests/test_robustez.py` cubriendo guards de optimize_k_node y `_ensure_api_key`).

### Pendiente (acordado con usuario, May 2026)
- Memoria entre corridas (persistencia + comparación) — diferido, nivel 2 del plan.
- Marco Metodológico — equipo metodológico lo construye en paralelo; cuando entreguen `methodology_v1.md`, inyectar en `INTERPRETATION_PROMPT` + `REFINEMENT_PROMPT` y ampliar `ArchetypeDescription` con tensión/motivadores/barreras/heurísticas/detonante/palanca.
- Pulido equivalente al paso 1 para los pasos 2 y 3 (próxima sesión — la base ya está sólida).
- Eventualmente nivel 3: SaaS multi-tenant (auth, persistencia DB, deployment, CI/CD) — fuera de scope ahora.

## Como ejecutar
- **Activar venv:** `source .venv/bin/activate`
- **Tests:** `python3 -m pytest tests/ -v` → 80/80
- **UI Streamlit:** `streamlit run src/ui/app.py`
- **Requisito:** configurar `OPENROUTER_API_KEY` en `.env` (usa `.env.example` como plantilla)

## Arquitectura
- Pipeline LangGraph: ingest → profile → **column_selection(LLM)** → preprocess(LLM) → optimize_k → select (determinístico, KMeans) → cluster → evaluate → interpret(LLM) → refinement(LLM)
- column_selection, preprocess, refinement, data_qa usan Claude Sonnet 4.5 vía OpenRouter
- interpret usa x-ai/grok-4.1-fast vía OpenRouter (modelo narrativo separado)
- UI Streamlit — 3 pasos (en español): Datos, Analizar, Arquetipos. Archivos en `src/ui/views/` (no `pages/` para evitar auto-multipage de Streamlit).
- Clustering: KMeans, AgglomerativeClustering (DBSCAN y GaussianMixture eliminados). Select fija KMeans.

## Notas técnicas de ingesta y preprocesamiento
- `ingest.py`: validación mínima 2 filas (antes 10) — soporta Excels con pestañas pequeñas. Mensajes en español.
- `preprocessor.py`: columnas categóricas se convierten a `str` antes de imputar — fix para Excels con tipos mixtos (int + str).
- `views/datos.py`: usa `pd.ExcelFile` para leer pestañas; si hay más de una muestra `selectbox`.
- `column_filter.py`: filtros estáticos deterministas. **Orden importante: datetime ANTES que id**, porque las fechas tienen cardinalidad alta y serían marcadas como ID. ID heuristic distingue numéricas continuas (income con todos únicos) de IDs (enteros secuenciales con `diffs == 1` o nombre matchea regex).
- `column_selection_node.py`: expone `suggest_columns(df, context)` como función pura para que la UI llame antes del pipeline; el nodo respeta `selected_columns` upstream del estado o, si no, usa la sugerencia LLM como selección final.

## Estructura UI (`src/ui/`)
- `app.py` — entry point, topbar con brand mark + wizard progress (3 segmentos) + nombre del paso actual + toggle Modo avanzado + chip del dataset.
- `styles.py` — CSS custom inyectado globalmente. Tokens documentados (COLOR/SPACING/TYPE), clases `.var-list`, `.space-*`, `.qh-grade--*`, `.wizard-progress`, `.hero-onboarding`, `.pipeline-checklist`, `.success-hero`. Scrollbar custom para containers.
- `copy.py` — microcopy centralizada con convenciones de tone of voice.
- `quality.py` — `silhouette_to_quality()`, `natural_log_message()`, `PIPELINE_UI_STEPS` (lista ordenada de 8 nodos visibles), `nodes_with_logs()` para mapear logs del grafo a la UI.
- `export.py` — `archetypes_to_csv`, `labels_to_csv`, `build_markdown_report`.
- `views/datos.py` — paso 1 single-page (donut tipos + contexto + variables; preview 5 filas + chat). `LOAD_ERROR_MAP` para errores de upload.
- `views/analizar.py` — paso 2: checklist en vivo de 8 pasos + try/except con retry + validación API key + panel ✨.
- `views/arquetipos.py` — paso 3: quality card + cluster sizes + cards compactas (con "Ver detalles") + tabs Mapa/Comparar/Por variable/**Conversar** + popover descargas + expander metodología.
- `components/` — `archetype_cards.py` (cards compactas con expander), `cluster_plots.py` (incluye `render_silhouette_curve`), `column_selector.py` (con tooltips importance + expander Recomendadas), `data_chat.py` (con scroll interno), `data_preview.py` (5 filas + `render_type_donut`), `profile_cards.py`.

## Marco Metodológico (pendiente de construir)
Objetivo: consolidar los múltiples documentos metodológicos existentes en un marco único que se inyecte en los prompts del pipeline (principalmente `interpret`, `refinement`) para que los arquetipos y narrativas reflejen la metodología propia del estudio.

**Secciones que debe tener el documento metodológico:**
1. **Fundamentos** — definición propia de "arquetipo" (qué es y qué no es), unidad de análisis.
2. **Marco teórico** — corrientes de ciencias del comportamiento en uso (COM-B, EAST, Fogg, Dual Process, SDT, etc.), glosario operativo.
3. **Schema del arquetipo** — campos fijos (nombre, tensión, motivadores, barreras, heurísticas, contexto detonante, palanca de cambio), longitud y tono por campo, ejemplos buenos/malos.
4. **Narrativa** — voz, tono, qué debe explicar (porqué conductual, teoría, implicación de diseño), formato.
5. **Criterios de calidad** — checklist de arquetipo bien formado, errores típicos (ej. confundir demografía con conducta).
6. **Ejemplos canónicos** — 2-3 arquetipos reales de proyectos pasados.
7. **Alcance** — qué decisiones no toma el arquetipo.

Documento completo en vault: `06 - archetypeSuite - Marco Metodológico.md`.

## Notas técnicas
- Python 3.9 — usar `typing.Dict`, `typing.List`, `typing.Optional` (no `dict | None` syntax).
- En `src/models/state.py` y `src/models/schemas.py` NO usar `from __future__ import annotations` (LangGraph y Pydantic v2 evalúan hints en runtime).
- Streamlit 1.50: `st.container(height=...)` soporta scroll interno — útil para listas largas dentro de cards.
- `st.expander` ejecuta los widgets internos siempre (sólo es display); el estado de checkboxes persiste aunque colapses.
- Dataset de ejemplo: `sample_data/customers.csv` (50 filas, 8 columnas).
- Los modelos vía OpenRouter a veces ignoran `response_format: json_object` y envuelven en markdown — usar siempre `extract_json()` al parsear respuestas LLM.
