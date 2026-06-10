"""Genera el dataset demo `bienestar_digital.csv` (comportamiento en redes sociales).

A diferencia de un sintético-uniforme (columnas independientes → silhouette ~0.06),
este planta 4 perfiles comportamentales con correlaciones realistas — más horas de
uso ↔ peor sueño ↔ menor bienestar — para que el pipeline encuentre arquetipos con
calidad Media/Alta, como corresponde a un demo honesto del producto.

Reproducible: python3 sample_data/generate_bienestar_digital.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# (nombre, proporción) — los nombres NO se exportan: el pipeline debe descubrirlos.
PROFILES = [
    ("scroll_nocturno", 0.26),
    ("uso_funcional", 0.30),
    ("creador_activo", 0.19),
    ("social_pasivo", 0.25),
]
N = 900


def _clip(a, lo, hi):
    return np.clip(a, lo, hi)


def make_profile(name: str, n: int) -> pd.DataFrame:
    if name == "scroll_nocturno":
        horas = _clip(rng.normal(7.0, 0.55, n), 5.4, 9.5)
        sesiones = _clip(rng.normal(17, 1.8, n), 12, 23).round()
        publicaciones = _clip(rng.normal(1.2, 1.0, n), 0, 5).round()
        interacciones = _clip(rng.normal(45, 9, n), 22, 80).round()
        bienestar = _clip(rng.normal(2.6, 0.55, n), 1, 4.2)
        edad = _clip(rng.normal(22, 3.5, n), 16, 34).round()
        franja = rng.choice(["Madrugada", "Noche", "Tarde"], n, p=[0.60, 0.37, 0.03])
        proposito = rng.choice(
            ["Entretenimiento", "Conexión social", "Noticias"], n, p=[0.85, 0.10, 0.05])
        sueno = rng.choice(
            ["Severo", "Moderado", "Leve"], n, p=[0.62, 0.32, 0.06])
        preocupacion = rng.choice(["Sí", "A veces", "No"], n, p=[0.65, 0.28, 0.07])
        pausas = rng.choice(["Nunca", "A veces"], n, p=[0.85, 0.15])
    elif name == "uso_funcional":
        horas = _clip(rng.normal(1.2, 0.3, n), 0.3, 2.0)
        sesiones = _clip(rng.normal(3.5, 1.0, n), 1, 6).round()
        publicaciones = _clip(rng.normal(0.8, 0.9, n), 0, 4).round()
        interacciones = _clip(rng.normal(10, 4, n), 2, 24).round()
        bienestar = _clip(rng.normal(8.2, 0.6, n), 6.5, 10)
        edad = _clip(rng.normal(41, 6, n), 28, 60).round()
        franja = rng.choice(["Mañana", "Tarde", "Noche"], n, p=[0.45, 0.45, 0.10])
        proposito = rng.choice(
            ["Trabajo", "Noticias", "Conexión social"], n, p=[0.60, 0.32, 0.08])
        sueno = rng.choice(["Sin impacto", "Leve"], n, p=[0.92, 0.08])
        preocupacion = rng.choice(["No", "A veces"], n, p=[0.82, 0.18])
        pausas = rng.choice(["Frecuentemente", "A veces"], n, p=[0.85, 0.15])
    elif name == "creador_activo":
        horas = _clip(rng.normal(5.2, 0.4, n), 4.0, 6.8)
        sesiones = _clip(rng.normal(6.5, 1.0, n), 4, 9).round()
        publicaciones = _clip(rng.normal(18, 3.0, n), 11, 30).round()
        interacciones = _clip(rng.normal(150, 18, n), 105, 220).round()
        bienestar = _clip(rng.normal(6.8, 0.6, n), 5.0, 8.8)
        edad = _clip(rng.normal(27, 4.5, n), 18, 40).round()
        franja = rng.choice(["Tarde", "Noche", "Mañana"], n, p=[0.72, 0.22, 0.06])
        proposito = rng.choice(
            ["Crear contenido", "Trabajo", "Entretenimiento"], n, p=[0.90, 0.06, 0.04])
        sueno = rng.choice(["Leve", "Moderado", "Sin impacto"], n, p=[0.55, 0.27, 0.18])
        preocupacion = rng.choice(["A veces", "Sí", "No"], n, p=[0.48, 0.30, 0.22])
        pausas = rng.choice(["A veces", "Nunca", "Frecuentemente"], n, p=[0.52, 0.28, 0.20])
    else:  # social_pasivo
        horas = _clip(rng.normal(2.9, 0.4, n), 1.8, 4.2)
        sesiones = _clip(rng.normal(12, 1.5, n), 9, 16).round()
        publicaciones = _clip(rng.normal(2.2, 1.3, n), 0, 7).round()
        interacciones = _clip(rng.normal(24, 6, n), 10, 45).round()
        bienestar = _clip(rng.normal(5.0, 0.6, n), 3.2, 6.8)
        edad = _clip(rng.normal(33, 5.5, n), 20, 50).round()
        franja = rng.choice(["Noche", "Tarde", "Mañana"], n, p=[0.60, 0.30, 0.10])
        proposito = rng.choice(
            ["Conexión social", "Entretenimiento", "Noticias"], n, p=[0.78, 0.16, 0.06])
        sueno = rng.choice(["Leve", "Moderado", "Sin impacto"], n, p=[0.55, 0.25, 0.20])
        preocupacion = rng.choice(["A veces", "No", "Sí"], n, p=[0.42, 0.34, 0.24])
        pausas = rng.choice(["A veces", "Nunca", "Frecuentemente"], n, p=[0.55, 0.30, 0.15])

    minutos_sesion = _clip(horas * 60 / np.maximum(sesiones, 1) + rng.normal(0, 4, n), 3, 120)
    return pd.DataFrame({
        "edad": edad.astype(int),
        "genero": rng.choice(["Femenino", "Masculino", "No binario"], n, p=[0.47, 0.47, 0.06]),
        "horas_uso_diario": horas.round(1),
        "sesiones_por_dia": sesiones.astype(int),
        "minutos_por_sesion": minutos_sesion.round(1),
        "franja_uso_principal": franja,
        "publicaciones_por_semana": publicaciones.astype(int),
        "interacciones_por_dia": interacciones.astype(int),
        "proposito_principal": proposito,
        "impacto_en_sueno": sueno,
        "bienestar_autoreportado": bienestar.round(1),
        "preocupacion_tiempo_pantalla": preocupacion,
        "toma_pausas": pausas,
        "dispositivo_principal": rng.choice(
            ["Celular", "Computador", "Tablet"], n, p=[0.78, 0.16, 0.06]),
    })


def main() -> None:
    parts = [make_profile(name, int(N * share)) for name, share in PROFILES]
    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # mezclar perfiles
    out = Path(__file__).parent / "bienestar_digital.csv"
    df.to_csv(out, index=False)
    print(f"{out.name}: {df.shape[0]} filas × {df.shape[1]} columnas")


if __name__ == "__main__":
    main()
