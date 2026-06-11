from io import StringIO
from typing import Any, Dict, List

import pandas as pd


def _join(items) -> str:
    return " | ".join(items or [])


def archetypes_to_csv(archetypes: List[Dict[str, Any]]) -> bytes:
    rows = []
    for a in archetypes:
        rows.append(
            {
                "cluster_id": a.get("cluster_id"),
                "label": a.get("label", ""),
                "nivel_cautela": a.get("nivel_cautela", ""),
                "description": a.get("description", ""),
                "comportamiento_principal": a.get("comportamiento_principal", ""),
                "microcomportamientos": _join(a.get("microcomportamientos")),
                "barreras": _join(a.get("barreras")),
                "habilitadores": _join(a.get("habilitadores")),
                "oportunidades_accion": _join(a.get("oportunidades_accion")),
                "cautela_reason": a.get("cautela_reason", ""),
                # Legacy (runs anteriores al marco comportamental)
                "key_characteristics": _join(a.get("key_characteristics")),
                "differentiators": _join(a.get("differentiators")),
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
            if a.get("nivel_cautela"):
                reason = a.get("cautela_reason", "")
                buf.write(f"*Nivel de cautela: **{a['nivel_cautela']}***{' — ' + reason if reason else ''}\n\n")
            if a.get("description"):
                buf.write(f"{a['description']}\n\n")
            if a.get("comportamiento_principal"):
                buf.write(f"**Comportamiento principal:** {a['comportamiento_principal']}\n\n")

            def _section(title: str, key: str):
                items = a.get(key, []) or []
                if items:
                    buf.write(f"**{title}:**\n\n")
                    for it in items:
                        buf.write(f"- {it}\n")
                    buf.write("\n")

            _section("Microcomportamientos", "microcomportamientos")
            _section("Barreras probables", "barreras")
            _section("Habilitadores", "habilitadores")
            _section("Oportunidades de acción", "oportunidades_accion")
            # Legacy
            _section("Características clave", "key_characteristics")
            _section("Diferenciadores", "differentiators")

    return buf.getvalue()
