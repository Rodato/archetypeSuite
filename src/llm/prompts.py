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
## Algoritmo Utilizado: {algorithm}
## Parámetros: {params}
## Iteración de Refinamiento: {refinement_count} de 2

## Contexto del Dataset
{context}

## Algoritmos Disponibles (SOLO puedes sugerir estos)
{algorithm_descriptions}

## Umbrales de Calidad
- Silhouette Score: > 0.25 es aceptable, > 0.5 es bueno
- Calinski-Harabasz: mayor es mejor (> 50 aceptable)
- Davies-Bouldin: menor es mejor (< 2.0 aceptable)

## Instrucciones
Decide si el clustering debe refinarse. Considera:
- ¿Son aceptables las métricas de calidad?
- ¿Un algoritmo o parámetros diferentes mejorarían los resultados?
- ¿Es probable que un refinamiento adicional ayude?
- Si sugieres un algoritmo, DEBE ser uno de los disponibles arriba.

Responde SOLO con un objeto JSON:
{{
  "should_refine": true/false,
  "reason": "Explicación",
  "suggested_algorithm": "NombreAlgoritmo o null",
  "suggested_params": {{...}} o null
}}
"""
