import numpy as np
import pandas as pd
import pytest

from src.data.column_filter import (
    apply_static_filters,
    detect_datetime,
    detect_free_text,
    detect_high_missing,
    detect_id_column,
    detect_near_zero_variance,
    extract_datetime_features,
)


def test_detect_id_by_unique_cardinality():
    df = pd.DataFrame({"foo": [1, 2, 3, 4, 5]})
    assert detect_id_column(df, "foo") is True


def test_detect_id_by_name_pattern():
    df = pd.DataFrame({"customer_id": [1, 1, 2, 2, 3], "user_uuid": ["a", "a", "b", "b", "c"]})
    assert detect_id_column(df, "customer_id") is True
    assert detect_id_column(df, "user_uuid") is True


def test_detect_id_negative():
    df = pd.DataFrame({"age": [25, 30, 25, 30, 25]})
    assert detect_id_column(df, "age") is False


def test_detect_near_zero_constant_column():
    df = pd.DataFrame({"flag": [1, 1, 1, 1, 1]})
    assert detect_near_zero_variance(df, "flag") is True


def test_detect_near_zero_categorical_single_value():
    df = pd.DataFrame({"status": ["active"] * 10})
    assert detect_near_zero_variance(df, "status") is True


def test_detect_near_zero_low_cv_numeric():
    df = pd.DataFrame({"price": [100.0, 100.001, 100.002, 100.0, 100.0005]})
    assert detect_near_zero_variance(df, "price") is True


def test_detect_near_zero_normal_numeric_negative():
    np.random.seed(0)
    df = pd.DataFrame({"income": np.random.randint(20000, 100000, 50)})
    assert detect_near_zero_variance(df, "income") is False


def test_detect_free_text_long_strings():
    df = pd.DataFrame({
        "comment": [
            "Esta es una opinión larga sobre el producto que escribió el cliente",
            "Otro comentario extenso sobre la experiencia general del usuario",
            "Mi feedback detallado acerca del servicio y atención recibida hoy",
        ]
    })
    assert detect_free_text(df, "comment") is True


def test_detect_free_text_short_categorical_negative():
    df = pd.DataFrame({"region": ["North", "South", "East", "West"] * 10})
    assert detect_free_text(df, "region") is False


def test_detect_free_text_high_unique_ratio_medium_length():
    df = pd.DataFrame({"name": [f"Persona Numero {i:02d}" for i in range(20)]})
    assert detect_free_text(df, "name") is True


def test_detect_high_missing():
    df = pd.DataFrame({"sparse": [None] * 8 + [1, 2]})
    assert detect_high_missing(df, "sparse") is True
    assert detect_high_missing(pd.DataFrame({"dense": list(range(10))}), "dense") is False


def test_detect_datetime_iso_strings():
    df = pd.DataFrame({"created_at": [
        "2024-01-15", "2024-02-20", "2024-03-25", "2024-04-30", "2024-05-12"
    ]})
    assert detect_datetime(df, "created_at") is True


def test_detect_datetime_native_dtype():
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=5)})
    assert detect_datetime(df, "ts") is True


def test_detect_datetime_negative_for_text():
    df = pd.DataFrame({"region": ["North", "South", "East"] * 5})
    assert detect_datetime(df, "region") is False


def test_extract_datetime_features_creates_year_column():
    df = pd.DataFrame({
        "created_at": ["2022-01-15", "2023-05-20", "2024-08-10", "2024-11-30"],
        "value": [10, 20, 30, 40],
    })
    out = extract_datetime_features(df, "created_at")
    assert "created_at" not in out.columns
    assert "created_at_year" in out.columns
    assert sorted(out["created_at_year"].dropna().unique().tolist()) == [2022.0, 2023.0, 2024.0]


def test_apply_static_filters_drops_id_constant_freetext_and_extracts_datetime():
    df = pd.DataFrame({
        "customer_id": list(range(20)),
        "country": ["MX"] * 20,
        "comment": [
            "Comentario muy largo del cliente sobre su experiencia con el producto"
        ] * 20,
        "age": np.random.RandomState(0).randint(20, 60, 20),
        "region": (["North", "South"] * 10),
        "signup_date": pd.date_range("2020-01-01", periods=20, freq="D").astype(str).tolist(),
    })

    filtered, report = apply_static_filters(df)

    dropped_cols = {d["column"] for d in report["dropped"]}
    assert "customer_id" in dropped_cols
    assert "country" in dropped_cols
    assert "comment" in dropped_cols

    assert "age" in filtered.columns
    assert "region" in filtered.columns

    assert "signup_date" not in filtered.columns
    assert "signup_date_year" in filtered.columns
    assert any(d["original"] == "signup_date" for d in report["datetime_extracted"])


def test_apply_static_filters_keeps_clean_dataset(sample_df):
    filtered, report = apply_static_filters(sample_df)
    assert set(filtered.columns) == set(sample_df.columns)
    assert report["dropped"] == []
    assert report["datetime_extracted"] == []
