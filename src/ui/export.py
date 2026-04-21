from io import StringIO
from typing import Any, Dict, List

import pandas as pd


def archetypes_to_csv(archetypes: List[Dict[str, Any]]) -> bytes:
    rows = []
    for a in archetypes:
        rows.append(
            {
                "cluster_id": a.get("cluster_id"),
                "label": a.get("label", ""),
                "description": a.get("description", ""),
                "key_characteristics": " | ".join(a.get("key_characteristics", []) or []),
                "differentiators": " | ".join(a.get("differentiators", []) or []),
            }
        )
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


def labels_to_csv(
    raw_data: Dict[str, list],
    labels: List[int],
    archetypes: List[Dict[str, Any]],
) -> bytes:
    df = pd.DataFrame(raw_data)
    df["Cluster"] = labels
    label_map = {a["cluster_id"]: a.get("label", f"Cluster {a['cluster_id']}") for a in archetypes}
    df["Arquetipo"] = df["Cluster"].map(label_map).fillna("Desconocido")
    return df.to_csv(index=False).encode("utf-8")


def build_markdown_report(result: Dict[str, Any]) -> str:
    buf = StringIO()
    buf.write("# Reporte de Arquetipos\n\n")

    file_name = result.get("file_name", "desconocido")
    buf.write(f"**Dataset:** {file_name}\n\n")

    context = result.get("dataset_context")
    if context:
        buf.write("## Contexto del dataset\n\n")
        buf.write(f"{context}\n\n")

    algo = result.get("selected_algorithm", "N/A")
    n_clusters = result.get("n_clusters", "N/A")
    refinement_count = result.get("refinement_count", 0)
    buf.write("## Configuración\n\n")
    buf.write(f"- Algoritmo: **{algo}**\n")
    buf.write(f"- Número de clusters: **{n_clusters}**\n")
    buf.write(f"- Iteraciones de refinamiento: **{refinement_count}**\n\n")

    metrics = result.get("metrics", {})
    if metrics:
        buf.write("## Métricas de clustering\n\n")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                buf.write(f"- {key}: {value:.3f}\n")
            else:
                buf.write(f"- {key}: {value}\n")
        buf.write("\n")

    archetypes = result.get("archetypes", [])
    if archetypes:
        buf.write("## Arquetipos\n\n")
        for a in archetypes:
            buf.write(f"### Cluster {a.get('cluster_id')}: {a.get('label', '')}\n\n")
            if a.get("description"):
                buf.write(f"{a['description']}\n\n")
            chars = a.get("key_characteristics", []) or []
            if chars:
                buf.write("**Características clave:**\n\n")
                for c in chars:
                    buf.write(f"- {c}\n")
                buf.write("\n")
            diffs = a.get("differentiators", []) or []
            if diffs:
                buf.write("**Diferenciadores:**\n\n")
                for d in diffs:
                    buf.write(f"- {d}\n")
                buf.write("\n")

    return buf.getvalue()
