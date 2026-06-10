# PLAN-LANZAMIENTO — Archetype Suite

> Plan ejecutable para cerrar el gap entre `descriptores-lanzamiento-plural.md` y el código real,
> rumbo al **video de lanzamiento** y a **beta pública**.
> Decisión de dirección: **el código alcanza al MD** (construir/activar, no recortar claims).
> Última actualización: **2026-06-10** — integra la auditoría completa de 4 capas
> (pipeline `src/` · API `api/` · front `web/` · tests/infra). Tracking por checkboxes.

---

## 1. Contraste claim (MD) → estado real (código)

| Claim del MD | Estado | Evidencia |
|---|---|---|
| **Wizard de 3 pasos** (Datos → Analizar → Arquetipos), dark mode, animaciones, español | **PARCIAL** — son **2 pasos** (`wizard-store.ts` step `1\|2`) + página de resultados `/runs/[id]` separada. Dark/animaciones/español: ✅ | `web/lib/wizard-store.ts:5`, `web/app/new/page.tsx`, `src/ui/copy.py` |
| Paso 1: drag&drop CSV/Excel + donut de tipos + contexto NL + **IA recomienda variables** con badges | **VERDAD** | `web/components/wizard/*`, `api/transform.py`, `src/models/schemas.py` (importance) |
| Paso 2: **checklist en vivo** de pipeline + SSE | **VERDAD** — muestra **8 pasos** (de 10 nodos reales; oculta `select`, `evaluate`) | `web/components/wizard/step-analizar.tsx`, `src/ui/quality.py`, `api/routers/analyze.py` |
| Paso 3: cards con **8 campos comportamentales** + nivel de cautela | **VERDAD** — `ArchetypeDescription` con los 8 campos | `src/models/schemas.py:38`, `web/components/results/archetype-card.tsx` |
| Exploración: **PCA + comparar + por variable + chat con gráficos** | **VERDAD** — chat decide si grafica (no siempre) | `web/app/runs/[id]/page.tsx`, `web/components/chat/data-chat.tsx` |
| **Exports**: CSV arquetipos + CSV etiquetado + Markdown | **VERDAD** | `api/routers/runs.py:90`, `src/ui/export.py` |
| Determinismo (T=0, `random_state=42`) · marco **COM-B + interseccionalidad** · nivel de cautela | **VERDAD** (con grietas — ver auditoría D1–D3) | `src/config/settings.py`, `knowledge_database/methodology_v1.md`, `src/agents/nodes/interpret_node.py` |
| Stack: Next 16, React 19, Tailwind 4, Base UI, Recharts, Framer, TanStack, Zustand · FastAPI SSE · LangGraph **10 nodos** · Sonnet 4.5 + Grok 4.3 · **sin BD, JSON en disco** | **VERDAD** | `web/package.json`, `pyproject.toml`, `src/agents/graph.py`, `api/store.py` |
| **108 tests passing** · falta auth/comparación/PDF/analytics | **VERDAD** | `tests/` (108 `def test_`) |

**Veredicto:** el MD es muy preciso. Gaps reales: (1) "wizard de 3 pasos" es 2+1 y conviene hacerlo literal,
(2) el chat no siempre grafica, (3) para beta faltan BD, auth, comparación, PDF, analytics (el propio MD ya lo
reconoce).

---

## 2. Auditoría completa (Jun 10, 2026) — hallazgos críticos

Cuatro auditorías en paralelo (pipeline, API, front, infra). La base es sólida: pipeline bien testeado,
contrato `lib/types.ts` ↔ `api/transform.py` fiel, cero `any` en TS, fallbacks LLM deterministas.
Los 5 hallazgos graves que NO estaban en este plan:

1. **Docker de producción corre SIN la metodología Plural.** `.dockerignore` excluye `knowledge_database/`
   y `Dockerfile.api` no la copia; `load_methodology()` es fail-soft → el contenedor usa un fallback de
   5 líneas en vez del marco de ~3.5k palabras. Degradación silenciosa del diferenciador del producto.
2. **Fuga de datos:** `api/transform.py:373` persiste `raw_data` completo en el JSON del run;
   `GET /api/runs/{id}` lo sirve a cualquiera; `GET /api/runs` enumera todos los IDs; `DELETE` sin auth;
   CORS `"*"` hardcodeado en `docker-compose.yml`.
3. **Upload sin defensas:** `file.read()` sin límite, parseo pandas dentro del event loop
   (`async def upload` congela el API), "hoja más rica" de Excel parsea TODAS las hojas (bomba xlsx),
   datasets en memoria nunca se desalojan.
4. **Determinismo con 3 grietas:** (a) `silhouette_score` O(n²) sin `sample_size` ×~10 llamadas —
   100k filas = horas; (b) `suggested_params` del LLM de refinement sin whitelist al constructor de
   KMeans (`refinement_node.py:43` → `executor.py:23`) — puede romper `random_state` o crashear tarde;
   (c) deps sin pin (venv ya resolvió pandas 3.0.3; `ingest.py:13` usa API privada de pandas).
   Bonus: filtro `contains` del chat sin `regex=False` (`data_qa.py:102`) → ReDoS.
5. **Todo el round SaaS (Jun 5) sin tests y sin CI:** `api/` (~920 líneas: SSE, store, transform,
   exports) cero tests; `web/` cero tests; no existe `.github/workflows`. El drift ya ocurrió
   (README/Makefile decían 92 tests con 108 reales).

Front (visible para usuarios): cambiar de tab en `/runs/[id]` **borra el chat** (Base UI desmonta
paneles, falta `keepMounted`); delete sin confirmación + caché stale; F5 en wizard pierde todo
(Zustand sin `persist`); dashboard muestra "Aún no tienes análisis" cuando el backend está caído.

Admin: la nota "Python 3.9" de CLAUDE.md está obsoleta — el venv YA es 3.11.14 y Docker usa 3.11-slim.

---

## 3. Fase 0 · Quick wins — ✅ COMPLETADA (Jun 10, 2026)

- [x] `keepMounted` en los 4 tabs de `/runs/[id]` — el chat y los charts sobreviven el cambio de tab.
- [x] `regex=False` en el filtro `contains` del chat (`src/llm/data_qa.py`) — cierra ReDoS.
- [x] Docker: `COPY knowledge_database/methodology_v1.md` en `Dockerfile.api` + excepción
      `!knowledge_database/methodology_v1.md` en `.dockerignore`. (El COPY explícito hace que un
      fallo sea ruidoso en build, no degradación silenciosa. Confirmar en el próximo
      `docker compose up --build` — Docker no estaba corriendo al cerrar la fase.)
- [x] CORS: compose ahora usa `${ARCHETYPE_CORS_ORIGINS:-http://localhost:3000}` (sin `"*"`).
- [x] `run_id` validado con regex `^[0-9a-f]{12}$` en `api/store.py` (`_run_path`).
- [x] Delete de run: Dialog de confirmación + `onError` con toast + `removeQueries(["run", id])`.
- [x] Invalidar `["runs"]` al evento `done` del SSE (`step-analizar.tsx`).
- [x] `requires-python = ">=3.11"` + nota Python de `CLAUDE.md` actualizada.
- [x] README/Makefile/CLAUDE.md: 92 → 108 tests.

**✅ Verificado:** `pytest` 108/108 + `tsc --noEmit` en verde tras todos los cambios.

## 4. Fase 1 · Cimientos de calidad — ✅ COMPLETADA (Jun 10, 2026)

- [x] **CI GitHub Actions** (`.github/workflows/ci.yml`): job backend (3.11, instala de
      `requirements.lock`, `pytest -q`) + job frontend (pnpm 11 + node 22, `tsc --noEmit`,
      `pnpm build`). `pnpm build` verificado en verde localmente antes de subir.
- [x] **`tests/test_api.py` con `TestClient`** — 24 tests: upload (ok/400) · sample · get 404 ·
      suggest (LLM mock) · chat dataset/run (mock) · analyze SSE completo (graph mock: progress→
      done→persistencia→list) · SSE error (sin API key / pipeline crash) · run_id anti-traversal ·
      exports ×3 · delete idempotente · unit `to_jsonable` + `build_run_record`/`run_summary`.
      **Bonus: encontró un bug real** — `pd.NaT` se serializaba como string `"NaT"` (fix en
      `api/serialization.py`).
- [x] **Lockfile Python**: `requirements.lock` (pip freeze del venv que pasa la suite, 92 pins —
      pandas 3.0.3, sklearn 1.8.0, langgraph 1.2.1...). `Dockerfile.api` y el CI instalan del lock.
      `requirements.txt` quedó como pointer (`-e .`) — una sola fuente de floors (pyproject).
- [x] **Whitelist de hiperparámetros** en `refinement_node` (`_sanitize_suggested_params`: solo
      `init` ∈ {k-means++, random}, `n_init`/`max_iter` int>0; descartes se loggean) + executor
      re-fuerza `random_state=settings.random_seed` si está presente. 3 tests nuevos.
- [x] API privada `STR_NA_VALUES` reemplazada por `_PANDAS_DEFAULT_NA` propia (set documentado
      de pandas). 1 test nuevo.

**✅ Verificado:** 137/137 tests (108 + 24 API + 5 robustez) · `tsc --noEmit` y `pnpm build` en
verde · workflow no-gitignored. Falta solo ver el primer run verde en GitHub tras el push.

## 5. Fase 2 · Demo-ready (ex Grupo A)

- [ ] **Persistir el wizard** (`zustand/persist` + sessionStorage: dataset_id, contexto, selección,
      sugerencia) y reset solo explícito (`/new?fresh=1` o acción del usuario) — hoy F5 o back pierde
      todo (`web/lib/wizard-store.ts:19`, `web/app/new/page.tsx:21-24`). Prerequisito de A1.
- [ ] **A1 · Wizard real de 3 pasos:** extender `step: 1 | 2` a `1 | 2 | 3` y presentar resultados
      como paso 3 dentro del shell del wizard (barra de 3 segmentos). Portar la narrativa
      "Paso N de 3 · {nombre}" del legacy (`src/ui/app.py`).
- [ ] **A2 · Dataset de ejemplo vistoso (cambio social):** reemplazar `sample_data/customers.csv`
      por una encuesta de cambio de comportamiento que produzca arquetipos alineados al pitch.
      Cablear en `/datasets/sample` (`api/routers/datasets.py`) y popover "cambiar archivo".
- [ ] **A3 · Chat que grafica comparativas:** afinar `DataQuery` + reglas duras en `src/llm/data_qa.py`
      para que "¿cuál arquetipo tiene más barreras?" devuelva bar chart fiable. Test en
      `tests/test_data_qa.py` que lo fije.
- [ ] Propagar el `detail` del backend en `streamAnalyze` (`web/lib/api.ts:127` descarta el body
      del error HTTP) + timeout/stall guard del SSE.
- [ ] `error.tsx` + `not-found.tsx` en español; dashboard distingue backend caído de "sin análisis"
      (`web/app/page.tsx:22` ignora `isError`).
- [ ] A4 (opcional) · Documentar la decisión 8 pasos visibles vs 10 nodos en `CLAUDE.md`.

**✅ Avanzamos cuando:** el guion demo pasa de corrido — subir dataset nuevo → wizard de 3 pasos sin
salir del shell (con un F5 a mitad sin perder nada) → "¿cuál arquetipo tiene más barreras?" responde
con gráfico.

## 6. Fase 3 · Pre-beta (ex Grupo B + seguridad/escala)

Seguridad y robustez (nuevo, de la auditoría):
- [ ] **Endurecer upload:** cap de bytes (rechazar `Content-Length` > N MB + cap al leer), pasar
      `upload` a `def` síncrono (threadpool) o `run_in_threadpool`, conteo de hojas Excel con
      `openpyxl read_only`/`nrows` (no parse completo), cap de filas (~100k), TTL + tope con
      desalojo en `_DatasetStore` (`api/store.py:34`).
- [ ] **Record listo para tenancy:** separar `raw_data` del documento servido (summary vs data;
      `GET /runs/{id}` no debe incluirlo), añadir `owner: None` + `schema_version: 1` YA al record
      (`api/transform.py:341`), paginación en `list_runs`. Hace mecánica la migración Postgres+Clerk.
- [ ] **Pipeline fuera del request/response:** persistir run en estado `running` al inicio
      (escritura atómica `tmp + os.replace`), ejecutar como tarea de fondo con semáforo (máx ~3,
      429 con copy amigable), SSE solo observa — sobrevive desconexiones y doble click
      (`api/routers/analyze.py:108`).
- [ ] `silhouette_score(..., sample_size=min(n, 10_000), random_state=settings.random_seed)` en
      `src/data/k_optimizer.py:35` y `src/clustering/evaluator.py:57` — viabiliza 100k filas.
- [ ] **Logging + request IDs** (hoy: cero logging en `api/`); errores genéricos al cliente en vez
      de `str(exc)` (`analyze.py:130`, `data_qa.py:386`); `logger.exception` en los except anchos.

Features beta (del plan original):
- [ ] **B1 · Persistencia en BD:** migrar runs JSON-en-disco → Postgres manteniendo la interfaz de
      `api/store.py` (swap sin tocar routers). Decidir si datasets en sesión van a BD/blob.
      Reformular la línea del MD "sin BD" al completar.
- [ ] **B2 · Auth + multi-tenancy:** Clerk (mismo proveedor que Plural Monitor) + scoping de
      `list_runs`/`get_run`/`delete` por owner. Provisional si la beta abre antes: API key compartida
      (header) al menos para `DELETE`.
- [ ] **B3 · Comparación entre corridas:** dos runs lado a lado (conteo, silhouette, radar
      superpuesto, diffs de tamaños). Reutilizar `transform.py`.
- [ ] **B4 · PDF export:** junto a CSV + Markdown (`build_markdown_report` → PDF). Resuelve también
      el pendiente de `chat_pendientes.md`.
- [ ] **B5 · Analytics de uso** (qué datasets, cuántos runs, errores). CI ya queda cubierto en Fase 1.

**✅ Avanzamos cuando:** checklist beta completa — auth bloquea y scopea por usuario; runs persisten
en BD tras reinicio; comparar dos runs muestra radar superpuesto; PDF descarga. Más los nuevos:
un upload de 500 MB recibe 413 sin tumbar el API; cerrar la pestaña a mitad de análisis no pierde
el run; `GET /runs/{id}` ya no expone `raw_data`.

---

## 7. Backlog de hallazgos menores (de la auditoría, no bloquean fases)

- `interpret_node` no valida que el LLM cubra exactamente `n_clusters` cluster_ids → "Desconocido"
  silencioso (`api/transform.py:151`). Validar + rellenar con `_fallback_interpretation`.
- `invoke_json_with_retry` reintenta 401/404 (inútil — incidente grok-4.1); peor caso >10 min por
  corrida con OpenRouter caído. No reintentar errores no recuperables.
- Round-trip DataFrame↔dict en cada nodo del grafo (costo + pérdida de dtypes en 100k×50).
- Resumen de columnas triplicado (`profiler.py` / `column_selection_node.py` / `data_qa.py`).
- `row_pct` con grupos NaN → porcentaje NaN (`data_qa.py:147` groupby con dropna default).
- El pipeline importa de `src/ui` (`interpret_node.py:8` → `src.ui.quality`): mover
  `quality.py`/`export.py` a `src/core/` con shims antes de borrar Streamlit.
- Multi-hoja Excel: el API devuelve `sheets`/`sheet` pero el front nuevo no los usa (regresión vs
  Streamlit) — `web/lib/types.ts` ni los declara.
- Front: suscripción al store completo re-renderiza el grid en cada tecla (`step-datos.tsx:23`);
  `INITIAL_STEPS` duplica `PIPELINE_UI_STEPS`; a11y (checkbox sin rol, chat sin `aria-live`);
  colores hex saltándose tokens en charts; `shadcn` CLI en dependencies; 3 rutas 100% client
  (mover dashboard/run a RSC cuando entre Clerk); chips de clarificación reusables tras responder.
- `load_sql` ejecuta connection string + SQL crudos (solo UI legacy) — dejar fuera del deploy.
- Health check expone `has_api_key` y modelos a anónimos; upload devuelve 200 (→201), DELETE 200
  con body (→204), sample faltante 500 (→config); sin `/api/v1`.
- Dockerfile sin `HEALTHCHECK` ni usuario no-root; compose sin `condition: service_healthy`.
- Heurística de k duplicada en conflicto (`optimize_k_node.py:25` n//10 vs `k_optimizer.py:23` n//5).
- Umbrales 0.25/0.5 hardcodeados en 3 sitios (`quality.py`, `prompts.py` ×2).
- Modernización opcional 3.11: `pyupgrade --py311-plus` (227 usos de `Dict[`/`List[`/`Optional[`).
  Mantener la regla de NO `from __future__ import annotations` en `state.py`/`schemas.py`.

## 8. Archivos críticos
`web/lib/wizard-store.ts` · `web/app/new/page.tsx` · `web/app/runs/[id]/page.tsx` ·
`web/components/wizard/step-analizar.tsx` · `web/components/chat/data-chat.tsx` ·
`web/components/ui/tabs.tsx` · `web/lib/api.ts` · `api/store.py` · `api/routers/runs.py` ·
`api/routers/datasets.py` · `api/routers/analyze.py` · `api/transform.py` · `src/llm/data_qa.py` ·
`src/agents/nodes/refinement_node.py` · `src/data/k_optimizer.py` · `src/ui/export.py` ·
`src/ui/quality.py` · `src/models/schemas.py` · `sample_data/customers.csv` · `pyproject.toml` ·
`Dockerfile.api` · `.dockerignore` · `docker-compose.yml` · `CLAUDE.md`.

## 9. Convenciones del repo
- Determinismo es un valor del producto: no introducir aleatoriedad (mantener `temperature=0`, `random_state=42`).
- La metodología se carga desde disco (`knowledge_database/methodology_v1.md`) — ajustes de marco son cero-código.
- Tests backend: `python3 -m pytest tests/ -v` (108). Typecheck front: `cd web && pnpm exec tsc --noEmit`.
- Dev: `make dev` (API :8000 + Next :3000) o `docker compose up --build`. Requiere `OPENROUTER_API_KEY` en `.env`.
- Python 3.11 (venv y Docker) — la nota vieja "3.9" quedó obsoleta.
- Actualizar la doc Obsidian del proyecto cuando cambien nodos/PipelineState/clustering/modelos/prompts/páginas.
