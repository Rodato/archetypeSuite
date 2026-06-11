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
- **Tests backend:** `python3 -m pytest tests/ -v` → 173/173 · **Typecheck front:** `cd web && pnpm exec tsc --noEmit`
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

## Auditoría 4 capas + Fase 0/1 del plan de lanzamiento (Jun 10, 2026)
Auditoría completa (pipeline/API/front/infra, ~45 hallazgos) consolidada en `PLAN-LANZAMIENTO.md`
(plan por fases con checkboxes — Fase 0 y Fase 1 completadas; Fase 2 = demo, Fase 3 = beta).
- **Fase 0 (quick wins):** `keepMounted` en tabs de `/runs/[id]` (el chat ya no se borra al cambiar
  de tab) · `regex=False` en filtro `contains` del chat (ReDoS) · `methodology_v1.md` ahora SÍ entra
  a la imagen Docker (antes el deploy corría con el fallback de 5 líneas) · CORS del compose sin `"*"` ·
  `run_id` validado con regex 12-hex en `api/store.py` · delete de run con dialog de confirmación +
  `onError` + `removeQueries` · invalidación de `["runs"]` al terminar análisis · `requires-python>=3.11`.
- **Fase 1 (cimientos):** CI en GitHub Actions (`.github/workflows/ci.yml`: pytest + tsc + next build) ·
  `tests/test_api.py` (24 tests, TestClient con grafo/LLM mockeados — encontró y arregló bug real:
  `pd.NaT` se serializaba como string `"NaT"`) · **`requirements.lock`** (pip freeze del venv testeado;
  Dockerfile.api y CI instalan del lock — `requirements.txt` quedó como pointer `-e .`) · whitelist de
  hiperparámetros en `refinement_node` (solo `init/n_init/max_iter`, el LLM ya no puede romper
  `random_state` ni crashear KMeans) + executor re-fuerza `random_state=settings.random_seed` ·
  `NA_TOKENS` ya no usa la API privada `pd.io.parsers.readers.STR_NA_VALUES`.
- **Fase 2 (demo-ready, mismo día):** wizard persistido en sessionStorage (`skipHydration` +
  rehidratación manual; reset solo con `/new?fresh=1`) · paso 3 real (`step: 1|2|3` + `lastRunId`,
  card "completado" al volver a `/new`) · chat comparativo SIEMPRE grafica (`_resolve_chart_type`
  fuerza bar si el LLM eligió table/none en op comparativa de 2-24 filas + regla dura en prompt) ·
  `streamAnalyze` propaga `detail` del backend + watchdog de stall 180s · `error.tsx`/`not-found.tsx`
  en español · dashboard y `/runs/[id]` distinguen backend caído de "no existe" · actions del CI
  en majors Node 24. **Falta de Fase 2:** dataset demo de cambio social (A2 — decisión de tema).
- **Limpieza (Jun 10, tarde):** eliminada la UI Streamlit completa (`src/ui/` — `quality.py`/`export.py`
  movidos a `src/core/`), `load_sql` (riesgo SSRF) + dep sqlalchemy, deps streamlit/plotly (lock pasa de
  92 a 71 paquetes — Docker más liviano), `AlgorithmSelection`, constantes muertas, `datetime_columns`
  del state, helpers no usados del registry, `customers.csv`, y el panel "Selección de algoritmo" del
  front (KMeans es fijo — no hay selección que explicar). Fixture e2e ahora sintética (no depende de CSV).
  Fix previo del mismo día: CSV con separador `;`/tab (export Excel es-*) se detecta automáticamente.
- **Tests: 145/145.**
- Pendientes priorizados en `PLAN-LANZAMIENTO.md` (Fases 2-3) y backlog de hallazgos menores ahí mismo.

## Mesa de trabajo: curación + perfilado a demanda (Jun 10, 2026 — tarde)
Tras la limpieza del legacy, dos features de producto (decisión del usuario: Fase 3 sigue en espera):
- **Curación de arquetipos:** `PATCH /api/runs/{id}/archetypes/{cluster_id}` con whitelist de campos
  narrativos (label, descripción, 8 campos; `nivel_cautela` NO editable — piso determinista) +
  `validated`/`curated_at`. El label se propaga a cluster_sizes/radar/scatter/box. Front: lápiz en
  cada card → dialog de edición + badge "✓ validado". Exports reflejan la versión curada.
- **Perfilado a demanda** (`src/llm/group_profile.py` + `POST /api/runs/{id}/profile-group`):
  descripción NL → filtros (LLM, `GroupFilterSpec`, duck-type con `_apply_filters` de data_qa) →
  subset determinista → stats grupo-vs-total → narrativa con metodología (`GROUP_PROFILE_PROMPT`,
  modelo narrativo) → `GroupProfileDescription` con **piso de cautela por tamaño de muestra**
  (`caution_floor_for_group_size`: <30 alta, <100 media). Se persiste en `record["custom_profiles"]`
  (con id/created_at/filters/n/share) y se puede borrar. Front: sección "Perfilar un grupo" en el
  run, cards reusan ArchetypeCard (color violeta, sin edición). La columna "Arquetipo" del df
  reconstruido permite perfilar por arquetipo ("los del arquetipo X que...").
  E2E real verificado: "quienes usan redes de madrugada y nunca toman pausas" → filtros correctos,
  117 filas, narrativa COM-B con cautela media.
- **Tests: 158/158.**

## Arquitectura de agentes — Paso 0 + Paso 1 (Jun 10, 2026 — noche)
Principio rector acordado: **dos capas** — los números son deterministas (pipeline, sin agentes);
el lenguaje/exploración es agéntico (chat, interpretación, perfilado), con outputs curables.
- **Paso 0 · k de dos regímenes** (`select_optimal_k` en `k_optimizer.py`): si la curva de
  silhouette es plana (max−min < `k_flat_curve_range`=0.03), los datos no distinguen ningún k y el
  argmax se arrastraba al tope del rango (caso real: estudiantes_portugal daba k=10 con curva
  0.131→0.153) → se elige el mejor k ≤ `k_flat_max_k`=4 ("pocos y trabajables") + flag
  `flat_k_curve` en k_analysis + copy honesta en la card "¿Por qué N arquetipos?". Con señal
  (demo: pico 0.37 en k=4) → argmax como siempre. De paso: el cap n//10 vs n//5 quedó unificado
  (n//10, dentro de KOptimizer).
- **Paso 1 · Chat agéntico** (`src/llm/chat_agent.py` + `chat_tools.py`): loop ReAct hand-rolled
  (sin prebuilt) con presupuesto duro (`agent_max_tool_calls`=5) y 4 tools DETERMINISTAS:
  `consultar_datos` (el executor whitelisteado de data_qa — hereda de DataQuery), `ver_esquema`,
  `ver_arquetipos`, `comparar_grupos` (nueva — resuelve el pendiente "comparar dos grupos lado a
  lado" de chat_pendientes.md). `get_agent_llm()` en provider (sin response_format — el agente
  alterna tool-calls y texto). Flag `settings.agentic_chat=True` + fallback fail-soft al one-shot
  (answer_chat). La tabla/gráfica de la ÚLTIMA tool-call acompaña la respuesta; `trace` de
  tool-calls viaja en el payload del chat (UI del "pensando…" = paso 3 pendiente).
  E2E real verificado: pregunta comparativa multi-paso → ver_arquetipos → comparar_grupos con
  label equivocado (grupo vacío) → **auto-corrección** con el label exacto → respuesta con tabla.
- Pendientes de la arquitectura: Paso 2 (intérprete con evidencia) y Paso 3 (streaming del trace
  en la UI). El refinement sigue candidato a degradarse a gate determinista.
- **Tests: 173/173.**

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
- UI única: Next.js + FastAPI (ver sección SaaS rewrite). La UI Streamlit legacy fue eliminada (Jun 10, 2026); `quality.py`/`export.py` viven en `src/core/`.
- Clustering: KMeans, AgglomerativeClustering (DBSCAN y GaussianMixture eliminados). Select fija KMeans.

## Notas técnicas de ingesta y preprocesamiento
- `ingest.py`: validación mínima 2 filas (antes 10) — soporta Excels con pestañas pequeñas. Mensajes en español.
- `preprocessor.py`: columnas categóricas se convierten a `str` antes de imputar — fix para Excels con tipos mixtos (int + str).
- `views/datos.py`: usa `pd.ExcelFile` para leer pestañas; si hay más de una muestra `selectbox`.
- `column_filter.py`: filtros estáticos deterministas. **Orden importante: datetime ANTES que id**, porque las fechas tienen cardinalidad alta y serían marcadas como ID. ID heuristic distingue numéricas continuas (income con todos únicos) de IDs (enteros secuenciales con `diffs == 1` o nombre matchea regex).
- `column_selection_node.py`: expone `suggest_columns(df, context)` como función pura para que la UI llame antes del pipeline; el nodo respeta `selected_columns` upstream del estado o, si no, usa la sugerencia LLM como selección final.

## `src/core/` (antes `src/ui/quality.py` y `src/ui/export.py`)
La UI Streamlit (`src/ui/`) fue **eliminada el Jun 10, 2026** (legacy sin uso). Los dos módulos
puros que el pipeline y el API necesitaban se movieron a `src/core/`:
- `core/quality.py` — `silhouette_to_quality()`, `caution_from_silhouette()` + `CAUTION_ORDER`/`CAUTION_META`, `natural_log_message()`, `PIPELINE_UI_STEPS` (8 nodos visibles), `nodes_with_logs()`.
- `core/export.py` — `archetypes_to_csv`, `labels_to_csv`, `build_markdown_report`.

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
- **Python 3.11** (venv 3.11.14 + `python:3.11-slim` en Docker; `requires-python = ">=3.11"` desde Jun 10, 2026). El código existente usa `typing.Dict`/`List`/`Optional` por herencia 3.9 — sintaxis moderna (`dict[...]`, `X | None`) es válida en código nuevo; modernización masiva pendiente (opcional, `pyupgrade --py311-plus`).
- En `src/models/state.py` y `src/models/schemas.py` NO usar `from __future__ import annotations` (LangGraph y Pydantic v2 evalúan hints en runtime).
- Dataset de ejemplo (demo del lanzamiento): `sample_data/bienestar_digital.csv` (900×14, español) — **generado** por `sample_data/generate_bienestar_digital.py` (seed 42) con 4 perfiles comportamentales plantados (scroll nocturno · uso funcional · creador activo · social pasivo) y correlaciones realistas (horas ↔ sueño ↔ bienestar). E2E verificado: el pipeline recupera los 4 arquetipos con calidad "Buena" (silhouette 0.37). `social_media_user_behavior.csv` (2000×34, Kaggle-style) quedó para pruebas drag&drop pero es sintético-uniforme SIN estructura de clusters (silhouette ~0.07) — no usarlo de demo. `customers.csv` (50×8) es el legacy original.
- Los modelos vía OpenRouter a veces ignoran `response_format: json_object` y envuelven en markdown — usar siempre `extract_json()` al parsear respuestas LLM.
- **Checklist del paso 2 muestra 8 pasos de los 10 nodos reales** (oculta `select` y `evaluate` — son internos y durarían <1s en pantalla). Decisión deliberada y coherente con el MD de lanzamiento; la lista visible vive en `PIPELINE_UI_STEPS` (`src/core/quality.py`) y se replica en `INITIAL_STEPS` de `step-analizar.tsx`.
