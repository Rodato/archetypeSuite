"""Evidencia diferenciadora por cluster — determinista, sin LLM.

Se inyecta al prompt de interpretación para que las narrativas citen cifras
reales ("58% usa de madrugada vs 19% global") en vez de escribir de memoria.
Vive en la capa determinista: mismos clusters → misma evidencia, siempre.
"""
from typing import List

import pandas as pd

MAX_NUMERIC_PER_CLUSTER = 5
MAX_CAT_PER_CLUSTER = 4
MIN_CAT_GAP = 0.05  # sobre-representación mínima (5 puntos) para citar una categoría


def cluster_evidence(df: pd.DataFrame, labels: List[int]) -> str:
    """Resumen compacto por cluster: diferenciadores numéricos (en σ del total)
    y categorías sobre-representadas frente al dataset completo."""
    if df.empty or len(labels) != len(df):
        return "(sin evidencia adicional)"

    work = df.copy()
    work["__cluster__"] = [int(l) for l in labels]
    numeric_cols = [c for c in df.select_dtypes(include="number").columns]
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    blocks: List[str] = []
    for cid in sorted(set(work["__cluster__"])):
        group = work[work["__cluster__"] == cid]
        share = len(group) / len(work) * 100
        lines = [f"### Cluster {cid} — {len(group)} filas ({share:.0f}% del total)"]

        diffs = []
        for col in numeric_cols:
            gm, tm, ts = group[col].mean(), df[col].mean(), df[col].std()
            if pd.notna(gm) and pd.notna(tm) and ts and ts > 0:
                z = (gm - tm) / ts
                diffs.append((abs(z), str(col), float(gm), float(tm), float(z)))
        # Orden por |σ| desc; desempate por nombre — salida 100% determinista.
        for _, col, gm, tm, z in sorted(diffs, key=lambda t: (-t[0], t[1]))[:MAX_NUMERIC_PER_CLUSTER]:
            lines.append(f"- {col}: {gm:.2f} vs {tm:.2f} global ({z:+.1f}σ)")

        cat_diffs = []
        for col in cat_cols:
            shares_g = group[col].value_counts(normalize=True)
            shares_t = df[col].value_counts(normalize=True)
            for val, sg in shares_g.head(6).items():
                st = float(shares_t.get(val, 0.0))
                gap = float(sg) - st
                if gap >= MIN_CAT_GAP:
                    cat_diffs.append((gap, str(col), str(val), float(sg), st))
        for gap, col, val, sg, st in sorted(cat_diffs, key=lambda t: (-t[0], t[1], t[2]))[:MAX_CAT_PER_CLUSTER]:
            lines.append(f"- {col} = {val}: {sg * 100:.0f}% vs {st * 100:.0f}% global")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) if blocks else "(sin evidencia adicional)"
