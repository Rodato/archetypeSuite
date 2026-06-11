# Archetype Suite

SaaS de **arquetipos comportamentales** (Estudio Plural): sube una encuesta (CSV/Excel),
un pipeline determinista encuentra grupos, y una capa LLM los narra como **hipótesis
comportamentales** en clave Plural (COM-B, cautela, lentes críticos). Wizard de 3 pasos +
chat agéntico + curación humana. Plan de trabajo y backlog: **`PLAN-LANZAMIENTO.md`**.

## Documentación (Obsidian)
Notas en: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Documentición codigo/archetypeSuite/`
Actualizar cuando cambien: nodos del pipeline, PipelineState, algoritmos de clustering, modelos LLM, prompts, páginas UI.
No actualizar por: bugfixes menores, ajustes de umbrales, cambios de copy.

## Principio rector: dos capas
- **Capa determinista (los números):** ingest → profile → column_filter → preprocess →
  optimize_k → cluster → evaluate. `temperature=0`, `random_state=42`, KMeans fijo, deps
  pinneadas (`requirements.lock`). Misma entrada → mismos clusters, siempre. Aquí NO van agentes.
- **Capa agéntica (el lenguaje):** chat, interpretación, perfilado. Outputs = hipótesis
  **curables** (el equipo edita/valida). La evidencia que consumen se calcula determinísticamente.

## Arquitectura
- **Pipeline LangGraph (10 nodos):** ingest → profile → column_selection(LLM) → preprocess(LLM)
  → optimize_k → select(KMeans fijo) → cluster → evaluate → interpret(LLM) → refinement(gate determinista).
  Nodos LLM con `invoke_json_with_retry` + fallback determinista. Refinamiento máx 2 iteraciones.
  - `optimize_k`: regla de dos regímenes (`select_optimal_k`): curva de silhouette plana
    (max−min < 0.03) → mejor k ≤ 4 + flag `flat_k_curve` (evita el k=10 por fragmentación);
    con pico real → argmax. Cap de k: n//10 (dentro de `KOptimizer`).
  - `interpret`: recibe **evidencia diferenciadora por cluster** calculada determinísticamente
    (`src/core/evidence.py`: σ vs total + categorías sobre-representadas) y DEBE citar cifras.
    Piso de cautela determinista por silhouette (`caution_from_silhouette` — solo sube, nunca baja).
  - `refinement`: **gate determinista** (sin LLM): silhouette < `refinement_silhouette_threshold`
    (0.25) en la primera pasada → 1 reintento con `n_init=30`; si no, termina. El executor
    re-fuerza `random_state` siempre.
- **Modelos (OpenRouter):** Claude Sonnet 4.5 (`llm_model`: selección, preproceso, chat agéntico,
  filtros de perfilado) · x-ai/grok-4.3 (`llm_narrative_model`: interpret, narrativa del chat,
  perfilado). Los modelos a veces envuelven JSON en markdown → usar siempre `extract_json()`.
- **Chat agéntico** (`src/llm/chat_agent.py` + `chat_tools.py`): loop ReAct hand-rolled,
  presupuesto `agent_max_tool_calls=5`, cierre forzado, fallback fail-soft al one-shot
  (`answer_data_question`). Tools DETERMINISTAS: `consultar_datos` (executor whitelisteado de
  data_qa), `ver_esquema`, `ver_arquetipos`, `comparar_grupos` (tabla lado a lado). Flag
  `settings.agentic_chat=True`. El `trace` de tool-calls viaja en el payload y la UI lo muestra
  colapsable ("🔧 esquema → comparación de grupos", con marca de auto-corrección).
- **Perfilado a demanda** (`src/llm/group_profile.py` + `POST /api/runs/{id}/profile-group`):
  grupo en lenguaje natural → filtros (LLM) → subset determinista → stats grupo-vs-total →
  hipótesis de 8 campos con **piso de cautela por tamaño de muestra** (<30 alta, <100 media).
  Persistido en `record["custom_profiles"]`, borrable. El df reconstruido incluye columna
  "Arquetipo" → se puede perfilar por arquetipo.
- **Curación humana** (`PATCH /api/runs/{id}/archetypes/{cluster_id}`): el equipo edita los campos
  narrativos y marca `validated` (badge ✓). `nivel_cautela` NO es editable. El label se propaga a
  cluster_sizes/radar/scatter/box; los exports reflejan la versión curada.
- **`api/` (FastAPI):** `main.py` (CORS+health) · `routers/datasets.py` (upload/sample/suggest/chat)
  · `routers/analyze.py` (SSE de `graph.stream`) · `routers/runs.py` (list/get/chat/curación/
  perfilado/exports/delete) · `transform.py` (build_run_record, charts, serialize) ·
  `serialization.py` (`to_jsonable`) · `store.py` (datasets en memoria + runs JSON en
  `api/_data/runs/`, `run_id` validado 12-hex anti-traversal).
- **`web/` (Next.js 16 App Router, TS):** Tailwind v4 + shadcn/Base UI (usa `render`, no
  `asChild`; Tabs con `keepMounted` para no desmontar el chat) + Recharts + TanStack Query +
  Zustand. Wizard de 3 pasos persistido en sessionStorage (`skipHydration` + rehidratación manual;
  reset solo con `/new?fresh=1`). Rutas: `/` · `/new` · `/runs/[id]`. `error.tsx`/`not-found.tsx`
  en español; dashboard y run distinguen backend caído de "no existe".
- **`src/core/`:** `quality.py` (calidad/cautela/PIPELINE_UI_STEPS — 8 pasos visibles de 10 nodos,
  oculta `select` y `evaluate` a propósito) · `export.py` (CSV/markdown) · `evidence.py`.

## Cómo ejecutar
- **Dev:** `make dev` (API :8000 + Next :3000) · o `docker compose up --build`. Requiere
  `OPENROUTER_API_KEY` en `.env` (plantilla en `.env.example`).
- **Tests backend:** `python3 -m pytest tests/ -v` → 180/180 · **Typecheck front:** `cd web && pnpm exec tsc --noEmit`
- **CI:** `.github/workflows/ci.yml` (pytest desde `requirements.lock` + tsc + next build) en cada push/PR.
- **Lockfile:** regenerar con `source .venv/bin/activate && pip freeze --exclude-editable > requirements.lock`
  (lo consumen Docker y CI — el determinismo depende de los pins).

## Notas técnicas vigentes
- **Python 3.11** (venv 3.11.14, Docker `python:3.11-slim`, `requires-python>=3.11`). El código
  legado usa `typing.Dict`/`List` — sintaxis moderna ok en código nuevo.
- En `src/models/state.py` y `schemas.py` NO usar `from __future__ import annotations`
  (LangGraph/Pydantic evalúan hints en runtime).
- Ingesta robusta (`src/data/ingest.py`): NA sentinels ES/LatAm + defaults de pandas inlined
  (sin API privada), CSV con separador `;`/tab detectado (export Excel es-*), multi-hoja Excel
  (elige la más rica), encodings con fallback, coerción object→numérico, validación mínima 2 filas.
- `column_filter.py`: orden importa — **datetime ANTES que id** (las fechas tienen cardinalidad alta).
- PCA desactivado por defecto (`enable_pca=False` — colapso 1-D inflaba el silhouette).
- Dataset demo: `sample_data/bienestar_digital.csv` (900×14, español, **generado** por
  `generate_bienestar_digital.py` seed 42 con 4 perfiles plantados — el pipeline los recupera con
  calidad "Buena" 0.37). `social_media_user_behavior.csv` es sintético-uniforme SIN estructura
  (silhouette ~0.07) — solo para pruebas drag&drop. Con encuestas reales espera silhouette
  0.10-0.18 y cautela alta: es el producto funcionando, no un bug.
- La UI Streamlit fue eliminada (Jun 10, 2026). No existe `src/ui/`.

## Marco metodológico Plural
`knowledge_database/methodology_v1.md` (~3.5k palabras, 10 secciones) se inyecta a los prompts de
interpretación y perfilado vía `src/llm/methodology.py` (lru_cache, fail-soft) — ajustes del equipo
son **cero-código** (editar el .md). Va dentro de la imagen Docker (COPY explícito — si falta, el
build falla ruidosamente). Schema de 8 campos en `ArchetypeDescription`; arquetipo = hipótesis
comportamental, no retrato de persona.

## Estado y pendientes
Ver **`PLAN-LANZAMIENTO.md`** (tracking por checkboxes): Fases 0-2 ✅ (quick wins, CI+tests+lock,
demo-ready) · Mesa de trabajo ✅ (curación + perfilado) · Arquitectura de agentes pasos 0-3 ✅ ·
**Fase 3 (pre-beta: hardening de upload, raw_data fuera del GET, Postgres, Clerk, comparación de
corridas, PDF) EN ESPERA** por decisión del usuario · backlog de hallazgos menores en §7.
Mejora anotada: streaming live del trace del agente (SSE del chat).

## Historial de rounds (detalle en git log)
- **May 6-7:** polish UI Streamlit + limpieza foundational (80 tests).
- **May 19:** chat conversacional — memoria 3 vueltas, clarificación absoluto/relativo, line+heatmap (92).
- **Jun 5:** capa comportamental Plural integrada (8 campos + cautela) + audit 38 fixes + **SaaS
  rewrite** Next.js+FastAPI (Streamlit pasa a legacy) + modelo narrativo → grok-4.3 (108).
- **Jun 10 (mañana):** auditoría 4 capas (~45 hallazgos) → Fase 0 quick wins (keepMounted, ReDoS,
  methodology en Docker, CORS, anti-traversal) + Fase 1 cimientos (CI, test_api 24, lock, whitelist
  refinement, fix pd.NaT) (137) → Fase 2 demo-ready (wizard persistido + 3 pasos, chat grafica
  comparativas, errores dignos, actions Node 24) (143) → dataset demo generado (A2) + fix CSV `;` (146).
- **Jun 10 (tarde):** limpieza legacy (−2.6k líneas, lock 92→71) + curación de arquetipos +
  perfilado a demanda (158) → **arquitectura de agentes** pasos 0-3: k dos regímenes, chat agéntico
  con tools deterministas, evidencia en interpret, traza en UI, refinement → gate determinista (180).
