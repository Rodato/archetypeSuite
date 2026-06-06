# Archetype Suite

## Documentación (Obsidian)
Notas en: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Documentición codigo/archetypeSuite/`
Actualizar cuando cambien: nodos del pipeline, PipelineState, algoritmos de clustering, modelos LLM, prompts, páginas UI.
No actualizar por: bugfixes menores, ajustes de umbrales, cambios de copy.

## Estado del Proyecto (May 19, 2026)

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

### Round de chat conversacional (May 19, 2026) — commit `89f1d02`
Foco: que el chat del paso 1 (y tab "Conversar" del paso 3) se sienta como hablar con alguien que recuerda y entiende matices, no como una calculadora one-shot.

**Memoria de conversación:**
- `answer_data_question` ahora recibe `history` (lista `[{role, text}, ...]`) y lo pasa al prompt como "Historial reciente de la conversación".
- `_format_history` recorta a las últimas 3 vueltas (6 entradas), trunca cada mensaje a 280 chars y devuelve placeholder explícito cuando está vacío.
- El prompt enseña al LLM a reconstruir filtros previos cuando aparece "y de esos", "ese grupo", "y ahora por…", a reusar la métrica si la pregunta nueva la omite, y a priorizar la pregunta nueva si contradice el historial.
- La UI (`data_chat.py`) compacta el `st.session_state` del historial a `{role, text}` antes de pasarlo al LLM vía `_history_for_llm`.

**Absoluto vs relativo (clarificación con chips):**
- `DataQuery` gana 4 campos: `normalize: "none" | "row_pct" | "total_pct"`, `needs_clarification: bool`, `clarification_question`, `clarification_options`.
- Cuando la pregunta es ambigua ("cuántos hombres por arquetipo", "distribución de género en cada cluster"), el LLM responde con `needs_clarification=true` + 3 chips fijos: **["Conteo absoluto", "% dentro de cada grupo", "% del total"]**. `answer_data_question` cortocircuita la ejecución y devuelve `DataQAResult.clarification`.
- La UI renderiza los 3 chips bajo el último mensaje con `_render_clarification_chips`; al click, se relanza la pregunta original concatenando "(en {opción})" como hint para el LLM.
- Si el LLM detecta que la pregunta YA es explícita ("en porcentaje", "% del total", "absoluto") elige el `normalize` correcto y NO pide clarificación.
- `_apply_normalize` en `data_qa.py` aplica el porcentaje a `value_counts` y `groupby_count`, renombra la columna `conteo` → `porcentaje` y actualiza el `y` del chart para que Plotly lo pinte correctamente.

**Más tipos de gráfica + reglas duras de elección:**
- `ChartType` ahora incluye `"line"` (eje X ordinal: rangos binned, fechas, deciles) y `"heatmap"` (matriz de correlación con 3+ variables).
- `_render_chart` añade `px.line(markers=True)` y `px.imshow(text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)` para el heatmap (se pasa la matriz cuadrada, no la versión reset_index).
- El prompt incluye un bloque "Reglas duras de elección" para que el LLM no use "bar" por inercia: `correlation` con 3+ cols → siempre heatmap; eje X ordinal → preferir line; `pie` solo con ≤5 segmentos, etc.

**Tests:** `tests/test_data_qa.py` pasa de 11 a 23 tests (12 nuevos cubriendo heatmap, normalize row_pct/total_pct, clarificación, line, format_history vacío/truncado/3-vueltas, prompt-receives-history).

**Total tests: 92/92.**

### Pendiente (acordado con usuario, May 2026)
- **Marco Metodológico v1 — escrito, en revisión del equipo (May 12, 2026).** Archivo: `knowledge_database/methodology_v1.md` (~3.5k palabras, 10 secciones). Construido a partir de `knowledge_database/Plural Ai - Enfoque narrativo.md`, `garaje/Ficha para Construcción de Arquetipos.md`, `garaje/Comunicación en [Plural].md` y `garaje/Modelo ccc Plural .md`. El doc redefine arquetipo como **hipótesis comportamental** (no retrato de persona), introduce schema de 8 campos (nombre · descripción patrón · comportamiento principal · microcomportamientos · barreras · habilitadores · oportunidades · nivel de cautela), marco teórico COM-B + socioecológico + sociología cultural + lentes (género/interseccionalidad/acción sin daño), tabla evitar/priorizar de voz Plural, ejemplos canónicos y 10 "reglas duras para el LLM" como TL;DR. **Próximos pasos cuando el equipo termine la revisión:** (1) aplicar ajustes al `.md`; (2) decidir si ampliar `ArchetypeDescription` (4 campos actuales) con los 8 campos del schema o conservar 4 y enriquecer narrativa por dentro; (3) actualizar `INTERPRETATION_PROMPT` y `REFINEMENT_PROMPT` para inyectar el `.md` como bloque de sistema (cargar desde disco al iniciar el nodo); (4) actualizar `archetype_cards.py` si cambia el schema; (5) actualizar tests del nodo interpret (fixture `_interpret_response()` en `tests/test_pipeline_e2e.py`).
- Memoria entre corridas (persistencia + comparación) — diferido, nivel 2 del plan.
- Pulido equivalente al paso 1 para los pasos 2 y 3 (próxima sesión — la base ya está sólida).
- Mejoras de chat ya entregadas (memoria de conversación, absoluto vs relativo, line + heatmap). Quedan pendientes en `chat_pendientes.md`: comparar dos grupos con tablas lado a lado y exportar resultado del chat a PDF/CSV.
- Eventualmente nivel 3: SaaS multi-tenant (auth, persistencia DB, deployment, CI/CD) — fuera de scope ahora.

## Como ejecutar
- **Stack SaaS (recomendado):** `make dev` (API FastAPI :8000 + Next.js :3000 juntos) · o `docker compose up --build` · o `cd web && pnpm dev` + `uvicorn api.main:app --reload --port 8000`. Detalle en `README.md`.
- **Activar venv:** `source .venv/bin/activate`
- **Tests backend:** `python3 -m pytest tests/ -v` → 92/92 · **Typecheck front:** `cd web && pnpm exec tsc --noEmit`
- **UI Streamlit (legacy, sigue funcionando):** `streamlit run src/ui/app.py`
- **Requisito:** configurar `OPENROUTER_API_KEY` en `.env` (usa `.env.example` como plantilla)

## Capa comportamental integrada + auditoría del sistema (Jun 5, 2026)
Round dirigido por el usuario: se **integró `methodology_v1.md`** al pipeline (esto **levanta** la antigua restricción "no tocar prompts ni schema") y se arregló una **auditoría de 38 hallazgos** (parte por parte: ingesta, coherencia, clustering).

**Capa comportamental (Plural):**
- `src/llm/methodology.py`: `load_methodology()` cacheado (lru_cache) y fail-soft, lee `knowledge_database/methodology_v1.md`.
- `ArchetypeDescription` ampliado a **8 campos**: `label` (nombre provisional), `description` (patrón), `comportamiento_principal`, `microcomportamientos[]`, `barreras[]` (con sub-nivel COM-B), `habilitadores[]`, `oportunidades_accion[]`, `nivel_cautela` (baja/media/alta) + `cautela_reason`. Se conservan `key_characteristics`/`differentiators` como **legacy** (compat con runs viejos).
- `INTERPRETATION_PROMPT` reescrito: inyecta el `.md` como bloque de sistema + las 10 reglas duras + voz Plural (patrones, no personas; lenguaje hipotético; nada de marketing). Recibe `{silhouette}`.
- **Cautela determinística**: `interpret_node` fuerza `nivel_cautela` al piso de `caution_from_silhouette()` (en `quality.py`): silhouette<0.25→alta, <0.5→media, resto baja. Solo SUBE, nunca baja la del LLM.
- UI: badge de cautela + 4 secciones comportamentales en las cards (Next.js `archetype-card.tsx` y Streamlit `archetype_cards.py`), fallback a legacy. Exports (CSV/markdown) incluyen los campos nuevos.

**Fixes del audit (38):** ingesta (NA sentinels extendidos, coerción object→numérico con miles/moneda, multi-hoja Excel, fallback de encoding) · coherencia (operación `missing_values` — antes el chat inventaba "N filas con valores faltantes"; narrativa fiel a la tabla; coerción de filtros numéricos; donut con `has_missing` booleano) · clustering (**PCA desactivado por defecto** `settings.enable_pca=False` — causaba colapso 1-D que inflaba el silhouette; alias `mode→most_frequent`; one-hot con `max_categories`; escalar solo numéricas continuas no dummies; `k_max` ~n/5; try/except en preprocess_node; std=None en clusters de 1 fila; high-cardinality en column_filter).
- **Tests: 108/108** (94 previos + test_methodology + test_ingest_robustness + tests de preprocesamiento nuevos).

## SaaS rewrite — Next.js + FastAPI (Jun 5, 2026)
Round grande: la UI principal pasó de Streamlit a **Next.js 16 + FastAPI**, manteniendo el pipeline `src/` intacto. Streamlit queda como legacy.
- **`api/` (FastAPI):** envuelve el pipeline. `main.py` (app+CORS+health), `routers/` (datasets: upload/sample/suggest-columns/chat · analyze: SSE streaming de `graph.stream` · runs: list/get/chat/export/delete), `transform.py` (donut, PCA scatter, radar normalizado, box quartiles, build_run_record, serialización de chat), `serialization.py` (`to_jsonable`: numpy/NaN→null), `store.py` (datasets en memoria + runs persistidos en `api/_data/runs/*.json`). NO toca la lógica del pipeline; reutiliza `quality.py`/`export.py`/`data_qa.py` (son puros, sin Streamlit).
- **`web/` (Next.js 16, App Router, TS):** Tailwind v4 + shadcn/ui (estilo base-nova → **Base UI**, usa `render` no `asChild`; Accordion usa `multiple` no `type`) + Recharts + framer-motion + TanStack Query + Zustand (wizard) + next-themes (dark). Tokens de marca portados (slate+indigo→violeta, `--chart-1..8` = BRAND_PALETTE) en `app/globals.css`. Rutas: `/` (dashboard + Mis análisis), `/new` (wizard 2 pasos con SSE), `/runs/[id]` (resultados). Charts custom: heatmap (grid CSS) y box-plot (SVG); el resto Recharts. Cliente API + SSE en `lib/api.ts`, tipos en `lib/types.ts`.
- **Deploy:** `Dockerfile.api` (backend) + `web/Dockerfile` (Next standalone) + `docker-compose.yml`. Front → Vercel (`web/`, env `NEXT_PUBLIC_API_URL`); backend → host de contenedores (volumen en `/data`).
- **Fix de modelo:** `x-ai/grok-4.1-fast` quedó **deprecado (404)** en OpenRouter → `settings.llm_narrative_model` ahora es **`x-ai/grok-4.3`** (cambio de config, no de prompt). Sin esto, interpret y la narrativa del chat caían al fallback determinista.
- **Scatter robusto:** si el preprocess reduce a <2 dims (PCA-1), el mapa se calcula desde las numéricas crudas estandarizadas (PCA-2). Antes (Streamlit) el mapa quedaba vacío en ese caso.

## Arquitectura
- Pipeline LangGraph: ingest → profile → **column_selection(LLM)** → preprocess(LLM) → optimize_k → select (determinístico, KMeans) → cluster → evaluate → interpret(LLM) → refinement(LLM)
- column_selection, preprocess, refinement, data_qa usan Claude Sonnet 4.5 vía OpenRouter
- interpret usa x-ai/grok-4.3 vía OpenRouter (modelo narrativo separado)
- UI principal: Next.js + FastAPI (ver sección SaaS rewrite). UI Streamlit legacy en `src/ui/views/`.
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
- `components/` — `archetype_cards.py` (cards compactas con expander), `cluster_plots.py` (incluye `render_silhouette_curve`), `column_selector.py` (con tooltips importance + expander Recomendadas), `data_chat.py` (scroll interno + chips de clarificación absoluto/relativo + renderers de line y heatmap + memoria de 3 vueltas vía `_history_for_llm`), `data_preview.py` (5 filas + `render_type_donut`), `profile_cards.py`.

## Marco Metodológico (v1 escrito — en revisión del equipo)
Objetivo: que las narrativas e interpretaciones del pipeline hablen en clave Plural (cambio comportamental con lentes críticos), no en clave marketing.

**Archivo:** `knowledge_database/methodology_v1.md` (May 12, 2026).

**Insumos consolidados (en local — NO versionados):** los docs de `knowledge_database/` distintos a `methodology_v1.md` quedan fuera de git vía `.gitignore` (pptx/xlsx pesados + originales). Para referencia, los insumos textuales que se usaron son:
- `Plural Ai - Enfoque narrativo.md` — insumo central (definición de arquetipo como hipótesis comportamental, schema de 8 campos, tabla evitar/priorizar, frases prohibidas/preferidas).
- `garaje/Ficha para Construcción de Arquetipos.md` — 12 dimensiones de la ficha (contexto, identidades, condiciones, tensiones, habilitadores, relatos, emociones, canales, acción sin daño).
- `garaje/Comunicación en [Plural].md` — voz Plural literal (clara, crítica, cuidadora, inclusiva, situada).
- `garaje/Modelo ccc Plural .md` + `Bases y Enfoques Modelo ccc.md` — COM-B (capacidades/motivaciones/oportunidades), modelo socioecológico feminista, sociología cultural, fases del cambio.

**Estructura final del `.md` (10 secciones, ~3.5k palabras):**
1. Fundamentos (qué es / qué no es / unidad de análisis / para qué sirve).
2. Marco teórico (COM-B, socioecológico, sociología cultural, lentes transversales, fases del cambio).
3. Glosario operativo (definiciones cortas y citables: barrera, habilitador, microcomportamiento, tensión, narrativa cultural, etc.).
4. Schema del arquetipo (8 campos con definición + longitud + tono + ejemplos bueno/malo cada uno).
5. Voz y tono (tabla evitar/priorizar literal, frases prohibidas/preferidas, convenciones de estilo).
6. Checklist de calidad.
7. Anti-patrones (errores típicos).
8. Ejemplos canónicos (2 bien escritos + 1 anti-ejemplo comentado).
9. Lo que queda fuera del alcance + cuándo frenar.
10. Reglas duras para el LLM (TL;DR ejecutable, 10 puntos).

**Estado (Jun 5, 2026): INTEGRADO al pipeline** por decisión del usuario — ver la sección "Capa comportamental integrada" arriba. La restricción "no tocar prompts ni schema" quedó levantada. El `.md` se inyecta en `INTERPRETATION_PROMPT` vía `src/llm/methodology.py` (carga desde disco), `ArchetypeDescription` tiene los 8 campos, y la UI los renderiza con badge de cautela. Como el doc se carga desde disco, ajustes de copy del equipo son **cero-código** (editar el `.md` y listo).

## Notas técnicas
- Python 3.9 — usar `typing.Dict`, `typing.List`, `typing.Optional` (no `dict | None` syntax).
- En `src/models/state.py` y `src/models/schemas.py` NO usar `from __future__ import annotations` (LangGraph y Pydantic v2 evalúan hints en runtime).
- Streamlit 1.50: `st.container(height=...)` soporta scroll interno — útil para listas largas dentro de cards.
- `st.expander` ejecuta los widgets internos siempre (sólo es display); el estado de checkboxes persiste aunque colapses.
- Dataset de ejemplo: `sample_data/customers.csv` (50 filas, 8 columnas).
- Los modelos vía OpenRouter a veces ignoran `response_format: json_object` y envuelven en markdown — usar siempre `extract_json()` al parsear respuestas LLM.
