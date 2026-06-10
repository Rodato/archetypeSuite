# Archetype Suite

Pipeline agéntico de clustering que descubre **arquetipos de comportamiento** a partir de datos
tabulares (CSV / Excel) y los describe en lenguaje natural — empaquetado como una app SaaS.

Wizard de 3 pasos: **Datos → Analizar → Arquetipos**, con dashboard, historial de análisis,
charts interactivos y un chat que entiende tus datos.

## Stack

| Capa | Tecnología |
|------|-----------|
| **Frontend** | Next.js 16 (App Router) · React 19 · TypeScript · Tailwind v4 · shadcn/ui · Recharts · framer-motion · TanStack Query |
| **Backend API** | FastAPI (JSON + SSE) envolviendo el pipeline existente |
| **Pipeline** | LangGraph (10 nodos) · scikit-learn (KMeans) |
| **LLM** | Claude Sonnet 4.5 (selección/preproceso/refinamiento/chat) + Grok 4.3 (narrativa) vía OpenRouter |

El pipeline es **determinístico** (`temperature=0`, `random_state=42`, KMeans fijo): la misma
entrada produce el mismo arquetipo.

> El frontend Streamlit original sigue disponible (`streamlit run src/ui/app.py`), pero la UI
> principal es la app Next.js en `web/`.

## Arquitectura

```
Next.js (web/)  ──fetch + SSE──►  FastAPI (api/)  ──►  LangGraph pipeline (src/)
  app/ · components/                routers/ · transform.py        nodes/ · clustering/
  Tailwind + shadcn + Recharts      serialization · store          (lógica intacta)
```

- `api/` reutiliza `src/` sin tocar la lógica del pipeline; solo añade transporte HTTP,
  serialización JSON-safe (numpy/NaN → null) y un store de sesiones (memoria) + runs (disco).
- Los análisis se persisten como JSON en `api/_data/runs/` → la pantalla **"Mis análisis"**
  sobrevive a reinicios sin necesidad de base de datos.

## Cómo correr (local)

Requisito: `OPENROUTER_API_KEY` en `.env` (usa `.env.example` como plantilla).

### Opción A — un comando con Docker

```bash
cp .env.example .env   # y pon tu OPENROUTER_API_KEY
docker compose up --build
# Frontend → http://localhost:3000   ·   API docs → http://localhost:8000/docs
```

### Opción B — make (dos procesos en local)

```bash
# 1. venv + deps (Python 3.11+) y deps del frontend
python3 -m venv .venv && source .venv/bin/activate
make install            # pip install -e ".[dev]"  +  cd web && pnpm install

# 2. levantar API + Web juntos (Ctrl-C detiene ambos)
make dev
```

### Opción C — manual

```bash
# Terminal 1 — backend
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000

# Terminal 2 — frontend
cd web && pnpm install && pnpm dev
```

Abre **http://localhost:3000** y pulsa *"Probar con datos de ejemplo"* para el flujo completo.

## Deploy

- **Frontend → Vercel.** Apunta el proyecto a `web/`. Define la env var
  `NEXT_PUBLIC_API_URL` con la URL pública de tu backend. `pnpm build` ya emite `standalone`.
- **Backend → cualquier host de contenedores** (Render, Railway, Fly, Cloud Run):
  `docker build -f Dockerfile.api -t archetype-api .`. Define `OPENROUTER_API_KEY` y
  `ARCHETYPE_CORS_ORIGINS` (la URL de tu frontend). Monta un volumen en `/data` para conservar
  el historial.

## Tests

```bash
make test            # o: python3 -m pytest tests/ -q   → 143 passed
cd web && pnpm exec tsc --noEmit   # typecheck del frontend
```

Los tests del pipeline están mockeados — no necesitan `OPENROUTER_API_KEY`.

## Estructura

```
src/        # Pipeline LangGraph (graph, nodes, clustering, data, llm, models) — sin cambios de lógica
api/        # FastAPI: main, routers/, transform.py, serialization.py, store.py
web/        # Next.js: app/ (rutas) · components/ (ui, charts, wizard, chat, results) · lib/
tests/      # 143 tests (pytest — pipeline + capa API)
sample_data/  customers.csv (demo, 50×8)
Dockerfile.api · web/Dockerfile · docker-compose.yml · Makefile
```

Documentación interna detallada en `CLAUDE.md`.
