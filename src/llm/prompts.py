NATURAL_ANSWER_PROMPT = """\
Eres un analista de datos respondiendo a un colega. Contesta la pregunta de forma natural y conversacional, mencionando los números clave del resultado.

## Pregunta original
{question}

## Resultado del cálculo
{result}

## Reglas
- Responde en 1-2 oraciones en español, como hablando con un colega.
- Menciona los números clave del resultado (no inventes).
- NO uses jerga técnica: nada de "operación", "filtro", "groupby", "agregación", "binning".
- NO empieces con "Se cuenta", "Se muestra", "Se calcula", "Se filtra" ni verbos pasivos en tercera persona.
- Si el resultado tiene varios grupos, menciona los 2-3 más relevantes (más altos, más bajos, o el contraste).
- Si hay 0 resultados, dilo claramente y explica brevemente por qué.

## Ejemplos
Pregunta: "cuántos hombres mayores de 22"
Resultado: filas encontradas = 504
✓ "Hay 504 hombres mayores de 22 años en tu dataset."

Pregunta: "distribución de edad por arquetipo"
Resultado: tabla con 4 arquetipos y edad promedio
✓ "El arquetipo Pioneros es el más joven (promedio 24 años) mientras que Tradicionales llega a 51."

Pregunta: "ingreso promedio por género"
Resultado: Male=4200, Female=3950, Other=4050
✓ "Los hombres ganan en promedio $4,200 — un poco más que las mujeres ($3,950) y el grupo Other ($4,050)."

Responde SOLO con la oración (o dos), sin comillas ni formato.
"""


DATA_QA_PROMPT = """\
Eres un analista de datos. El usuario te hizo una pregunta en español sobre un \
dataset. Tu trabajo es elegir UNA operación pandas válida que conteste la pregunta, \
y describir el resultado esperado en una narrativa breve. NO ejecutes código — \
sólo devuelve la operación estructurada.

## Modo: {mode}
{mode_description}

## Contexto del dataset
{context}

## Columnas disponibles (con tipos y muestras)
{columns_summary}

## Pregunta del usuario
{question}

## Operaciones disponibles
- "describe": estadísticas descriptivas (mean, std, min, max…) de columnas numéricas. `columns` es lista.
- "value_counts": frecuencia de valores únicos. `columns` debe tener exactamente 1 columna.
- "groupby_count": cuenta filas por grupo. `groupby` lista de 1-2 columnas.
- "groupby_agg": agrega una o varias columnas numéricas (`columns`, lista) usando `agg`, agrupado por `groupby` (lista de 1-2 columnas). Para comparar varios indicadores entre grupos al mismo tiempo, pasa todas las columnas relevantes en `columns`.
- "distribution": histograma o boxplot de una columna numérica. `columns` debe tener 1 columna.
- "correlation": matriz de correlación entre columnas numéricas. `columns` lista de 2+ columnas numéricas.
- "top_n": top N valores de `columns[0]` ordenados por sí mismos (descendente). Requiere `top_n` (entero).
- "filter_count": cuenta filas que cumplen los filtros de `filter_by`. Usar para "¿cuántos X tienen Y?" sin agrupar. Si no hay filtros devuelve el total.

## Filtros — filter_by (opcional)
Para preguntas con condiciones (ej. "mayores de 25", "de género Male", "con ingresos > 50000"):
- Cada condición: `{{"column": "<col>", "op": "<op>", "value": <value>}}`
- Operadores disponibles: "eq" (igual), "ne" (distinto), "gt" (>), "lt" (<), "gte" (>=), "lte" (<=), "in" (lista de valores), "contains" (substring en texto)
- Para strings usa el valor tal como aparece en `top_values` (case-insensitive al ejecutar).
- Múltiples condiciones se combinan con AND.
- Ejemplo: hombres mayores de 22 → `"filter_by": [{{"column": "gender", "op": "eq", "value": "Male"}}, {{"column": "age", "op": "gt", "value": 22}}]`
- `filter_by` también aplica a `groupby_count`, `value_counts`, `groupby_agg`, etc. — filtra el dataframe antes de la operación.

## Rangos personalizados (bins) — opcional
Si la pregunta pide agrupar una columna numérica en rangos (ej. "edad de 16 a 19 y de 20 a 25", "ingresos bajos/medios/altos"), añade `bins`:
- Cada bin tiene `column` (la columna numérica original), `edges` (lista creciente de cortes inclusivos por la izquierda, ej. [16, 19, 25]) y `labels` opcional (una etiqueta por intervalo, ej. ["16-19", "20-25"]).
- La columna binned reemplaza a la original en `groupby`/`columns`. Sigue usando el nombre original — el binning se aplica antes de la operación.
- Si no se piden rangos, omite `bins` o pásalo como null.

## Tipos de gráfico
- "bar": barras (default para conteos y agregados)
- "pie": pastel (sólo para distribuciones de pocas categorías)
- "histogram": histograma (para distribución de numéricas)
- "box": boxplot (para distribución de numéricas, idealmente agrupada)
- "scatter": dispersión (para correlación entre 2 numéricas)
- "table": tabla (default si no aplica visualización)
- "none": sólo narrativa, sin tabla ni gráfico

## REGLAS DURAS
- Las columnas en `columns` y `groupby` DEBEN existir EXACTAMENTE como aparecen arriba.
- No inventes columnas. Si la pregunta no se puede responder con las columnas disponibles, devuelve `operation="filter_count"`, narrativa explicando la limitación, y `chart_type="none"`.
- La narrativa debe ser 1-2 oraciones EN ESPAÑOL describiendo qué calcula la operación (no el resultado, que aún no conoces).

Responde SOLO con un objeto JSON:
{{
  "operation": "<operation>",
  "columns": ["<col>", ...],
  "groupby": null o ["<col>", ...],
  "agg": null o "mean|median|sum|min|max|count",
  "top_n": null o <entero>,
  "bins": null o [{{"column": "<col>", "edges": [<n>, <n>, ...], "labels": null o ["<label>", ...]}}],
  "filter_by": null o [{{"column": "<col>", "op": "<op>", "value": <value>}}],
  "chart_type": "<chart_type>",
  "narrative": "Frase breve en español describiendo qué se calculó."
}}
"""

COLUMN_RELEVANCE_PROMPT = """\
Eres un experto en segmentación de clientes y ciencia del comportamiento. Dado un \
conjunto de datos y el objetivo del usuario, decide qué columnas son MÁS relevantes \
para construir arquetipos significativos y cuáles deberían quedar fuera.

## Contexto del dataset (objetivo del usuario)
{context}

## Columnas disponibles (ya filtradas — sin IDs, fechas, ni texto libre)
{columns_summary}

## Criterios para seleccionar
- Privilegia variables de COMPORTAMIENTO sobre demografía pura cuando ambas existan.
- Descarta columnas redundantes (correlación obvia o medidas equivalentes).
- Descarta columnas que el contexto del usuario sugiere que NO son relevantes para su objetivo.
- Si dudas entre una columna y otra, marca la menos clara con importancia "low" pero inclúyela.
- Mantén entre 4 y 10 columnas seleccionadas idealmente. Si hay menos de 4, marca todas como seleccionadas.

## Importancia
- "high": esencial para diferenciar arquetipos.
- "medium": aporta separación pero no es crítica.
- "low": útil sólo si el dataset es pequeño en variables.

## IMPORTANTE
- Los nombres de columna en `selected_columns` y `excluded_columns` DEBEN ser EXACTAMENTE iguales a los listados arriba (case-sensitive).
- No inventes nombres.

Responde SOLO con un objeto JSON:
{{
  "selected_columns": [
    {{"name": "<nombre_columna>", "reason": "<por qué es relevante>", "importance": "high|medium|low"}}
  ],
  "excluded_columns": [
    {{"name": "<nombre_columna>", "reason": "<por qué se excluye>"}}
  ],
  "summary": "Razonamiento general de la selección (1-2 oraciones)"
}}
"""

PREPROCESSING_PROMPT = """\
Eres un experto en ciencia de datos. Analiza el siguiente perfil de datos y decide \
la estrategia de preprocesamiento óptima para clustering.

## Perfil de Datos
{profile}

## Contexto del Dataset
{context}

## Instrucciones
Decide:
1. Qué columnas eliminar (IDs, constantes, texto de alta cardinalidad, etc.)
2. Estrategia de imputación: "mean", "median" o "mode"
3. Método de escalado: "standard", "minmax" o "robust"
4. Codificación para categóricas: "onehot" o "label"
5. Reducción de dimensionalidad: null o {{"method": "pca", "n_components": <número o fracción>}}

Responde SOLO con un objeto JSON:
{{
  "drop_columns": [...],
  "imputation": "...",
  "scaling": "...",
  "encoding": "...",
  "dimensionality_reduction": null o {{"method": "pca", "n_components": ...}},
  "reasoning": "Breve explicación de tus decisiones"
}}
"""

ALGORITHM_SELECTION_PROMPT = """\
Eres un experto en clustering. Basándote en las características de los datos, selecciona \
el mejor algoritmo de clustering y sus hiperparámetros.

## Características de los Datos
- Dimensiones: {n_rows} filas x {n_cols} columnas
- Columnas numéricas: {n_numeric}
- Preprocesamiento aplicado: {preprocessing_metadata}

## Contexto del Dataset
{context}

## Número Óptimo de Clusters (análisis matemático)
- Silhouette Analysis: k={best_silhouette_k} (score={best_silhouette_score:.3f})
- Elbow Method: k={elbow_k}
- **k seleccionado: {optimal_k}** — DEBES usar este valor en tus parámetros.

## Algoritmos Disponibles
{algorithm_descriptions}

## Instrucciones
Selecciona el algoritmo y especifica sus hiperparámetros. El número de clusters \
ya fue determinado matemáticamente: usa n_clusters={optimal_k}.

Responde SOLO con un objeto JSON:
{{
  "algorithm": "NombreAlgoritmo",
  "params": {{...}},
  "reasoning": "Breve explicación"
}}
"""

INTERPRETATION_PROMPT = """\
Eres un experto en análisis de negocio y marketing. Analiza los resultados del clustering \
y crea descripciones de arquetipos vívidas y accionables. Responde siempre en español.

## Perfiles de Clusters (estadísticas por cluster usando columnas ORIGINALES)
{cluster_profiles}

## Métricas de Clustering
{metrics}

## Número de Clusters: {n_clusters}

## Columnas Originales: {original_columns}

## Contexto del Dataset
{context}

## Instrucciones
Para cada cluster, crea un arquetipo con:
- Un nombre memorable (ej: "Familias Ahorrativas", "Millennials Tecnológicos")
- Una descripción narrativa de 2-3 oraciones
- 3-5 características clave que definen el arquetipo
- 2-3 diferenciadores (qué los hace únicos vs los otros clusters)

Responde SOLO con un objeto JSON:
{{
  "archetypes": [
    {{
      "cluster_id": 0,
      "label": "...",
      "description": "...",
      "key_characteristics": ["...", "..."],
      "differentiators": ["...", "..."]
    }}
  ],
  "summary": "Resumen general de la segmentación"
}}
"""

REFINEMENT_PROMPT = """\
Eres un experto en calidad de clustering. Revisa los resultados y decide \
si se necesita refinamiento.

## Métricas Actuales
{metrics}

## Número de Clusters: {n_clusters}
## Algoritmo Utilizado: {algorithm} (fijo — no puede cambiarse)
## Parámetros: {params}
## Iteración de Refinamiento: {refinement_count} de 2

## Contexto del Dataset
{context}

## Umbrales de Calidad
- Silhouette Score: > 0.25 es aceptable, > 0.5 es bueno
- Calinski-Harabasz: mayor es mejor (> 50 aceptable)
- Davies-Bouldin: menor es mejor (< 2.0 aceptable)

## Instrucciones
Decide si el clustering debe refinarse. El algoritmo y el número de clusters \
están fijos — solo puedes sugerir ajustes de hiperparámetros secundarios \
(ej. `init`, `n_init`, `max_iter`). No sugieras cambios de `n_clusters` \
ni de algoritmo.

Considera:
- ¿Son aceptables las métricas de calidad?
- ¿Ajustar hiperparámetros mejoraría los resultados?
- ¿Es probable que un refinamiento adicional ayude?

Responde SOLO con un objeto JSON:
{{
  "should_refine": true/false,
  "reason": "Explicación",
  "suggested_algorithm": null,
  "suggested_params": {{...}} o null
}}
"""
