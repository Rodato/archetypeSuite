# Archetype Suite

## Estado del Proyecto (Feb 19, 2026)
- UI probada con datos reales por primera vez — pipeline funciona end-to-end
- UI y prompts traducidos completamente al español
- Contexto de dataset añadido (campo opcional en la página de carga)
- Tests: 33/33 (pendiente re-verificar tras cambios de hoy)

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
- Clustering: KMeans, DBSCAN, AgglomerativeClustering, GaussianMixture

## Cambios realizados Feb 19, 2026
1. **Modelo narrativo separado**: `interpret_node` usa `x-ai/grok-4.1-fast` en vez de Claude Sonnet
   - `src/config/settings.py`: nuevo campo `llm_narrative_model`
   - `src/llm/provider.py`: nueva función `get_narrative_llm()`
   - `src/agents/nodes/interpret_node.py`: usa `get_narrative_llm()`

2. **Fix parsing LLM**: Claude y Grok devuelven JSON envuelto en markdown ```json ... ```
   - `src/llm/provider.py`: función `extract_json()` que limpia el markdown
   - Los 4 nodos LLM usan `extract_json()` antes de `json.loads()`

3. **Fix Grok archetypes**: Grok metía "summary" como elemento del array archetypes
   - `src/agents/nodes/interpret_node.py`: filtra elementos no-dict del array antes de parsear

4. **UI en español**: todas las páginas y componentes traducidos
   - `src/ui/app.py`, todas las páginas en `src/ui/pages/`, todos los componentes en `src/ui/components/`

5. **Contexto de dataset**: campo opcional que el usuario llena al cargar datos
   - `src/models/state.py`: nuevo campo `dataset_context: Optional[str]`
   - `src/ui/pages/upload.py`: textarea con placeholder descriptivo
   - `src/ui/pages/pipeline.py`: pasa `dataset_context` al estado inicial
   - `src/llm/prompts.py`: todos los prompts traducidos al español con bloque `{context}`
   - Los 4 nodos LLM leen y pasan el contexto al prompt

## Notas tecnicas
- Python 3.9 - se usa `typing.Dict`, `typing.List`, `typing.Optional` en vez de `dict | None` syntax
- En `src/models/state.py` NO usar `from __future__ import annotations` porque LangGraph evalua type hints en runtime
- En `src/models/schemas.py` (Pydantic) tampoco usar `from __future__ import annotations`
- Dataset de ejemplo: `sample_data/customers.csv` (50 filas, 8 columnas)
- Los modelos via OpenRouter a veces ignoran `response_format: json_object` y envuelven en markdown — usar siempre `extract_json()` al parsear respuestas LLM
