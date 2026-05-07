# Archetype Suite

Pipeline agéntico de clustering que descubre arquetipos de comportamiento a partir de datos tabulares (CSV / Excel) y los describe en lenguaje natural.

Wizard de 3 pasos: **Datos → Analizar → Arquetipos**.

## Stack

- **LangGraph** — orquesta un pipeline de 10 nodos (ingest, profile, column_selection, preprocess, optimize_k, select, cluster, evaluate, interpret, refinement).
- **scikit-learn** — clustering (KMeans / Agglomerative).
- **Streamlit** — UI.
- **Claude Sonnet 4.5** + **Grok 4.1 Fast** vía OpenRouter — selección de columnas, preprocesamiento, interpretación y refinamiento.

El pipeline es determinístico (`temperature=0`, `random_state=42`, KMeans fijo), así que la misma entrada produce el mismo arquetipo.

## Setup

```bash
# 1. Clonar y entrar
git clone <repo>
cd archetypeSuite

# 2. Crear venv (Python 3.11+ recomendado)
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar dependencias
pip install -e .
# o si vas a desplegar en Streamlit Cloud:
pip install -r requirements.txt

# 4. Configurar API key
cp .env.example .env
# Editar .env y poner tu OPENROUTER_API_KEY (https://openrouter.ai/keys)
```

## Cómo correr

```bash
# UI en localhost:8501
streamlit run src/ui/app.py

# Tests
python3 -m pytest tests/ -v
```

Hay un dataset de ejemplo en `sample_data/customers.csv` accesible desde el botón "Probar con datos de ejemplo" en el paso 1.

## Estructura

```
src/
  agents/        # Pipeline LangGraph (graph, routing, nodes/)
  clustering/    # Algoritmos (registry, executor, evaluator)
  config/        # Settings (pydantic-settings desde .env)
  data/          # Ingesta, profiling, preprocesado, k-optimizer
  llm/           # Provider OpenRouter, prompts, helpers JSON
  models/        # Schemas Pydantic + PipelineState (TypedDict)
  ui/            # Streamlit (app.py + views/ + components/ + styles.py)
tests/           # 74 tests (pytest)
sample_data/     # Datasets de demo
```

## Notas para devs

- El proyecto usa `requires-python>=3.9` (`typing.Dict/List/Optional` en lugar de syntax `|`).
- No usar `from __future__ import annotations` en `state.py` ni `schemas.py` (LangGraph y Pydantic v2 evalúan hints en runtime).
- Los modelos LLM a veces envuelven JSON en markdown — usar siempre `extract_json()` (`src/llm/provider.py`) al parsear.
- Documentación interna detallada en `CLAUDE.md`.
