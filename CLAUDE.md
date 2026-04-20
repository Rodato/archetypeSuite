# Archetype Suite

## Documentación (Obsidian)
Notas en: `~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Documentición codigo/archetypeSuite/`
Actualizar cuando cambien: nodos del pipeline, PipelineState, algoritmos de clustering, modelos LLM, prompts, páginas UI.
No actualizar por: bugfixes menores, ajustes de umbrales, cambios de copy.

## Estado del Proyecto (Apr 20, 2026)
- UI probada con datos reales — pipeline funciona end-to-end
- UI y prompts completamente en español
- Contexto de dataset añadido (campo opcional en la página de carga)
- Soporte Excel multi-pestaña: selectbox automático si hay más de una pestaña
- Tests: 40/40

## Como ejecutar
- **Activar venv:** `source .venv/bin/activate`
- **Tests:** `python3 -m pytest tests/ -v`
- **UI Streamlit:** `streamlit run src/ui/app.py`
- **Requisito:** configurar `OPENROUTER_API_KEY` en `.env` antes de ejecutar el pipeline

## Arquitectura
- Pipeline LangGraph: ingest -> profile -> preprocess(LLM) -> select_algorithm(LLM) -> cluster -> evaluate -> interpret(LLM) -> refinement_check(LLM)
- preprocess, select, refinement usan Claude Sonnet 4.5 via OpenRouter
- interpret usa x-ai/grok-4.1-fast via OpenRouter (modelo narrativo separado)
- Loop de refinamiento limitado a 2 iteraciones max
- UI Streamlit con 5 páginas (en español): Cargar Datos, Perfil de Datos, Ejecutar Pipeline, Resultados, Explorar
- Clustering: KMeans, AgglomerativeClustering (DBSCAN y GaussianMixture eliminados)

## Notas técnicas de ingesta y preprocesamiento
- `ingest.py`: validación mínima 2 filas (antes 10) — soporta Excels con pestañas pequeñas
- `preprocessor.py`: columnas categóricas se convierten a `str` antes de imputar — fix para Excels con tipos mixtos (int + str en la misma columna)
- `upload.py`: usa `pd.ExcelFile` para leer pestañas; si hay más de una muestra `selectbox`

## Notas tecnicas
- Python 3.9 - se usa `typing.Dict`, `typing.List`, `typing.Optional` en vez de `dict | None` syntax
- En `src/models/state.py` NO usar `from __future__ import annotations` porque LangGraph evalua type hints en runtime
- En `src/models/schemas.py` (Pydantic) tampoco usar `from __future__ import annotations`
- Dataset de ejemplo: `sample_data/customers.csv` (50 filas, 8 columnas)
- Los modelos via OpenRouter a veces ignoran `response_format: json_object` y envuelven en markdown — usar siempre `extract_json()` al parsear respuestas LLM
