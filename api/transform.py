"""Pure transforms that turn pipeline output into frontend-ready, JSON-safe shapes.

Everything here mirrors the logic the Streamlit components used (cluster_plots.py,
data_preview.py, data_chat.py) so the Next.js app renders the exact same analysis — the
only difference is charts are computed here and drawn client-side instead of via Plotly.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from api.serialization import dataframe_to_table, to_jsonable
from src.llm.data_qa import DataQAResult
from src.core.quality import silhouette_to_quality

# Categorical color cycle — ported verbatim from cluster_plots.BRAND_PALETTE.
BRAND_PALETTE = [
    "#4F46E5", "#F59E0B", "#10B981", "#EF4444",
    "#8B5CF6", "#EC4899", "#14B8A6", "#F97316",
]

# Donut type colors — ported from data_preview.TYPE_COLORS.
TYPE_COLORS = {
    "Numéricas": "#4F46E5",
    "Categóricas": "#F59E0B",
    "Fechas": "#10B981",
    "Texto libre": "#94A3B8",
}


# --------------------------------------------------------------------------- #
# Upload-screen helpers
# --------------------------------------------------------------------------- #
def classify_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """4-category type breakdown for the donut + missing-data percentage.

    Mirrors data_preview._classify_columns exactly (incl. the >50%-unique free-text rule).
    """
    counts = {"Numéricas": 0, "Categóricas": 0, "Fechas": 0, "Texto libre": 0}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            counts["Categóricas"] += 1  # a boolean is a 2-category variable
        elif pd.api.types.is_numeric_dtype(s):
            counts["Numéricas"] += 1
        elif pd.api.types.is_datetime64_any_dtype(s):
            counts["Fechas"] += 1
        else:
            non_null = s.dropna()
            if len(non_null) > 0 and non_null.nunique() / len(non_null) > 0.5:
                counts["Texto libre"] += 1
            else:
                counts["Categóricas"] += 1

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = int(df.isna().sum().sum())
    missing_pct = (missing_cells / total_cells * 100) if total_cells else 0.0

    segments = [
        {"label": label, "value": value, "color": TYPE_COLORS[label]}
        for label, value in counts.items()
        if value > 0
    ]
    return {
        "segments": segments,
        "n_cols": int(df.shape[1]),
        "n_rows": int(df.shape[0]),
        "missing_pct": round(float(missing_pct), 1),
        "missing_cells": missing_cells,
        "has_missing": missing_cells > 0,  # truthful flag — not subject to rounding
    }


def preview_rows(df: pd.DataFrame, n: int = 5) -> Dict[str, Any]:
    return dataframe_to_table(df.head(n))


# --------------------------------------------------------------------------- #
# Results-screen charts
# --------------------------------------------------------------------------- #
def label_map_from_archetypes(archetypes: List[Dict[str, Any]]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for a in archetypes:
        try:
            out[int(a["cluster_id"])] = a.get("label", f"Arquetipo {a['cluster_id']}")
        except (KeyError, TypeError, ValueError):
            continue
    return out


def cluster_sizes_list(metrics: Dict[str, Any], label_map: Dict[int, str]) -> List[Dict[str, Any]]:
    sizes = metrics.get("cluster_sizes", {}) or {}
    out = []
    for raw_id, size in sizes.items():
        try:
            cid = int(raw_id)
        except (TypeError, ValueError):
            continue
        out.append({
            "cluster_id": cid,
            "label": label_map.get(cid, f"Arquetipo {cid}"),
            "size": int(size),
            "color": BRAND_PALETTE[cid % len(BRAND_PALETTE)],
        })
    out.sort(key=lambda r: r["cluster_id"])
    return out


def cluster_scatter(
    processed_data: Dict[str, list],
    raw_df: pd.DataFrame,
    numeric_cols: List[str],
    labels: List[int],
    label_map: Dict[int, str],
) -> List[Dict[str, Any]]:
    """2D projection coloured by archetype, for the "Mapa" tab.

    Prefers the processed feature matrix (faithful to the space clusters formed in). If the
    preprocess step reduced it to <2 dimensions, falls back to a PCA of the standardized raw
    numeric columns so the map is always meaningful.
    """
    if not labels:
        return []

    source: Optional[np.ndarray] = None
    proc_df = pd.DataFrame(processed_data) if processed_data else pd.DataFrame()
    if proc_df.shape[1] >= 2 and proc_df.shape[0] == len(labels):
        source = proc_df.fillna(0.0).to_numpy(dtype=float)
    elif numeric_cols and raw_df.shape[0] == len(labels):
        num = raw_df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        if num.shape[1] >= 2:
            num = num.fillna(num.mean()).fillna(0.0)
            source = StandardScaler().fit_transform(num.to_numpy(dtype=float))
    if source is None:
        return []

    if source.shape[1] > 2:
        coords = PCA(n_components=2, random_state=42).fit_transform(source)
    else:
        coords = source[:, :2]

    rows = []
    for (pc1, pc2), lab in zip(coords, labels):
        cid = int(lab)
        rows.append({
            "PC1": round(float(pc1), 4),
            "PC2": round(float(pc2), 4),
            "cluster_id": cid,
            "archetype": label_map.get(cid, "Desconocido"),
        })
    return rows


def radar_data(
    cluster_profiles: Dict[Any, Any],
    numeric_cols: List[str],
    label_map: Dict[int, str],
) -> Dict[str, Any]:
    """Per-cluster means, min-max normalized per column to [0,1] (mirror render_radar_chart)."""
    axes = numeric_cols[:8]
    if not axes or not cluster_profiles:
        return {"axes": [], "series": []}

    cluster_ids = sorted(int(k) for k in cluster_profiles.keys())

    # Raw means per cluster per axis.
    raw: Dict[int, List[float]] = {}
    for cid in cluster_ids:
        prof = cluster_profiles.get(cid) or cluster_profiles.get(str(cid)) or {}
        vals = []
        for col in axes:
            stat = prof.get(col) or {}
            mean = stat.get("mean")
            vals.append(float(mean) if mean is not None else 0.0)
        raw[cid] = vals

    # Per-axis min-max normalization across clusters.
    series = []
    for cid in cluster_ids:
        norm_vals = []
        for j in range(len(axes)):
            col_vals = [raw[c][j] for c in cluster_ids]
            lo, hi = min(col_vals), max(col_vals)
            rng = hi - lo
            norm_vals.append(round((raw[cid][j] - lo) / rng, 4) if rng else 0.5)
        series.append({
            "cluster_id": cid,
            "label": label_map.get(cid, f"Arquetipo {cid}"),
            "color": BRAND_PALETTE[cid % len(BRAND_PALETTE)],
            "values": norm_vals,
            "raw_values": [round(v, 3) for v in raw[cid]],
        })
    return {"axes": axes, "series": series}


def box_stats(
    raw_df: pd.DataFrame,
    labels: List[int],
    label_map: Dict[int, str],
    numeric_cols: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Precompute box-plot quartiles per numeric column per archetype."""
    df = raw_df.copy()
    df["__cluster__"] = labels
    out: Dict[str, List[Dict[str, Any]]] = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        groups = []
        for cid in sorted(set(int(l) for l in labels)):
            series = pd.to_numeric(df.loc[df["__cluster__"] == cid, col], errors="coerce").dropna()
            if series.empty:
                continue
            groups.append({
                "cluster_id": cid,
                "label": label_map.get(cid, f"Arquetipo {cid}"),
                "color": BRAND_PALETTE[cid % len(BRAND_PALETTE)],
                "min": round(float(series.min()), 3),
                "q1": round(float(series.quantile(0.25)), 3),
                "median": round(float(series.median()), 3),
                "q3": round(float(series.quantile(0.75)), 3),
                "max": round(float(series.max()), 3),
            })
        if groups:
            out[col] = groups
    return out


# --------------------------------------------------------------------------- #
# Chat result serialization
# --------------------------------------------------------------------------- #
def serialize_qa_result(result: DataQAResult) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "narrative": result.narrative,
        "operation": result.operation,
        "error": result.error,
        "clarification": to_jsonable(result.clarification) if result.clarification else None,
        "table": None,
        "chart": None,
    }
    if result.table is not None and not result.table.empty:
        payload["table"] = dataframe_to_table(result.table)
    if result.chart is not None:
        payload["chart"] = _serialize_chart(result.chart)
    return payload


def _serialize_chart(chart: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ctype = chart.get("type")
    data = chart.get("data")
    if ctype in (None, "table", "none") or data is None or getattr(data, "empty", False):
        return None

    if ctype == "heatmap":
        # `data` is a square correlation matrix (do NOT reset_index).
        matrix = data
        return {
            "type": "heatmap",
            "x_labels": [str(c) for c in matrix.columns],
            "y_labels": [str(i) for i in matrix.index],
            "z": [[_safe(v) for v in row] for row in matrix.to_numpy().tolist()],
            "x": None, "y": None, "color": None,
        }

    return {
        "type": ctype,
        "x": chart.get("x"),
        "y": chart.get("y"),
        "color": chart.get("color"),
        "data": dataframe_to_table(data),
    }


def _safe(v):
    try:
        f = float(v)
        return round(f, 3) if f == f else None  # f != f → NaN
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Run record assembly
# --------------------------------------------------------------------------- #
def build_run_record(
    final_state: Dict[str, Any],
    *,
    run_id: str,
    created_at: str,
    file_name: str,
    dataset_context: str,
) -> Dict[str, Any]:
    """Turn the final PipelineState into the persisted, JSON-safe run document."""
    archetypes = final_state.get("archetypes", []) or []
    metrics = final_state.get("metrics", {}) or {}
    cluster_profiles = final_state.get("cluster_profiles", {}) or {}
    labels = [int(x) for x in (final_state.get("labels", []) or [])]
    raw_data = final_state.get("raw_data", {}) or {}
    processed_data = final_state.get("processed_data", {}) or {}
    k_analysis = final_state.get("k_analysis", {}) or {}

    label_map = label_map_from_archetypes(archetypes)
    raw_df = pd.DataFrame(raw_data)
    numeric_cols = [c for c in raw_df.select_dtypes(include=np.number).columns.tolist()]
    categorical_cols = [c for c in raw_df.columns if c not in numeric_cols]

    silhouette = metrics.get("silhouette_score")
    quality = silhouette_to_quality(silhouette)
    quality = {**quality, "score": _safe(silhouette)}

    optimal_k = final_state.get("optimal_k") or final_state.get("n_clusters") or len(archetypes)
    total = sum(int(s) for s in (metrics.get("cluster_sizes", {}) or {}).values()) or len(labels)

    archetype_cards = []
    for a in archetypes:
        cid = int(a.get("cluster_id", 0))
        size = int((metrics.get("cluster_sizes", {}) or {}).get(cid,
                   (metrics.get("cluster_sizes", {}) or {}).get(str(cid), 0)))
        archetype_cards.append({
            "cluster_id": cid,
            "label": a.get("label", f"Arquetipo {cid}"),
            "description": a.get("description", ""),
            # Behavioral layer (methodology §4) — default-safe for old runs.
            "comportamiento_principal": a.get("comportamiento_principal", ""),
            "microcomportamientos": a.get("microcomportamientos", []) or [],
            "barreras": a.get("barreras", []) or [],
            "habilitadores": a.get("habilitadores", []) or [],
            "oportunidades_accion": a.get("oportunidades_accion", []) or [],
            "nivel_cautela": a.get("nivel_cautela", "media"),
            "cautela_reason": a.get("cautela_reason", ""),
            # Legacy fields (kept for backward compat with pre-methodology runs).
            "key_characteristics": a.get("key_characteristics", []) or [],
            "differentiators": a.get("differentiators", []) or [],
            "size": size,
            "prevalence": round(size / total * 100, 1) if total else 0.0,
            "color": BRAND_PALETTE[cid % len(BRAND_PALETTE)],
        })

    record = {
        "id": run_id,
        "created_at": created_at,
        "file_name": file_name,
        "dataset_context": dataset_context,
        "n_rows": int(raw_df.shape[0]),
        "n_cols": int(raw_df.shape[1]),
        "optimal_k": int(optimal_k),
        "n_clusters": int(metrics.get("n_clusters", len(archetypes))),
        "quality": quality,
        "archetypes": archetype_cards,
        "summary": final_state.get("interpretation_summary", ""),
        "cluster_sizes": cluster_sizes_list(metrics, label_map),
        "metrics": {
            "silhouette_score": _safe(metrics.get("silhouette_score")),
            "calinski_harabasz_score": _safe(metrics.get("calinski_harabasz_score")),
            "davies_bouldin_score": _safe(metrics.get("davies_bouldin_score")),
            "n_clusters": int(metrics.get("n_clusters", len(archetypes))),
        },
        "k_analysis": {
            "k_range": [int(k) for k in (k_analysis.get("k_range") or k_analysis.get("k_values") or [])],
            "silhouette_scores": [_safe(s) for s in (k_analysis.get("silhouette_scores") or [])],
            "optimal_k": int(k_analysis.get("optimal_k", optimal_k)) if k_analysis else int(optimal_k),
            "forced_k_min": bool(k_analysis.get("forced_k_min", False)),
        },
        "charts": {
            "scatter": cluster_scatter(processed_data, raw_df, numeric_cols, labels, label_map),
            "radar": radar_data(cluster_profiles, numeric_cols, label_map),
            "box": box_stats(raw_df, labels, label_map, numeric_cols),
        },
        "columns": {"numeric": numeric_cols, "categorical": categorical_cols},
        "labels": labels,
        "raw_data": to_jsonable(raw_data),
        "advanced": {
            "selected_algorithm": final_state.get("selected_algorithm", "KMeans"),
            "selection_reasoning": final_state.get("selection_reasoning", ""),
            "refinement_reason": final_state.get("refinement_reason", ""),
            "refinement_count": int(final_state.get("refinement_count", 0)),
        },
    }
    return to_jsonable(record)


def run_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight projection for the dashboard list. Defensive: a partial/old run file
    must never 500 the whole list."""
    quality = record.get("quality") or {}
    archetypes = record.get("archetypes") or []
    return {
        "id": record.get("id", ""),
        "created_at": record.get("created_at", ""),
        "file_name": record.get("file_name", "análisis"),
        "dataset_context": record.get("dataset_context", ""),
        "n_rows": record.get("n_rows", 0),
        "n_cols": record.get("n_cols", 0),
        "n_archetypes": len(archetypes),
        "optimal_k": record.get("optimal_k"),
        "quality": {
            "grade": quality.get("grade", "—"),
            "label": quality.get("label", "Sin calcular"),
            "color": quality.get("color", "gray"),
            "score": quality.get("score"),
        },
        "archetype_labels": [a.get("label", "") for a in archetypes],
    }
