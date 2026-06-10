"""Tests de la capa FastAPI (`api/`) — TestClient sobre la app real.

LLM y grafo mockeados: ningún test necesita OPENROUTER_API_KEY ni red.
Cubre el round-trip completo: upload → suggest → analyze (SSE) → get run →
exports → chat → delete, más unit tests de serialization y build_run_record.
"""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.store as store
from api.main import app
from api.serialization import to_jsonable
from api.transform import build_run_record, run_summary
from src.config.settings import settings
from src.llm.data_qa import DataQAResult

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

CSV_BYTES = (
    "edad,ingreso,ciudad\n"
    + "\n".join(
        f"{20 + i},{1000 + i * 10},{'Bogota' if i % 2 else 'Lima'}" for i in range(12)
    )
).encode("utf-8")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setattr(store, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(store.datasets, "_data", {})
    monkeypatch.setattr(settings, "openrouter_api_key", "test-key")
    return TestClient(app)


def make_final_state() -> dict:
    """PipelineState final mínimo pero realista (schema de 8 campos + métricas)."""
    base_archetype = {
        "cluster_id": 0,
        "label": "Exploradores prudentes",
        "description": "Patrón de evaluación cuidadosa antes de adoptar.",
        "comportamiento_principal": "Evalúan opciones antes de actuar",
        "microcomportamientos": ["Comparan precios", "Consultan a pares"],
        "barreras": ["Desconfianza institucional (motivación)"],
        "habilitadores": ["Redes de pares"],
        "oportunidades_accion": ["Pilotos comunitarios"],
        "nivel_cautela": "media",
        "cautela_reason": "Silhouette moderado",
        "key_characteristics": [],
        "differentiators": [],
    }
    return {
        "raw_data": {
            "edad": [25, 30, 35, 40, 45, 50],
            "ingreso": [1000, 1200, 3000, 3200, 5000, 5200],
            "ciudad": ["Lima", "Lima", "Bogota", "Bogota", "Quito", "Quito"],
        },
        "processed_data": {
            "f1": [0.1, 0.2, 0.8, 0.9, 1.5, 1.6],
            "f2": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        },
        "labels": [0, 0, 1, 1, 0, 1],
        "archetypes": [
            base_archetype,
            {**base_archetype, "cluster_id": 1, "label": "Adoptantes tempranos"},
        ],
        "interpretation_summary": "Dos grupos diferenciados.",
        "metrics": {
            "silhouette_score": 0.42,
            "calinski_harabasz_score": 10.5,
            "davies_bouldin_score": 0.8,
            "n_clusters": 2,
            "cluster_sizes": {0: 3, 1: 3},
        },
        "cluster_profiles": {
            0: {"edad": {"mean": 33.3}, "ingreso": {"mean": 2400.0}},
            1: {"edad": {"mean": 41.6}, "ingreso": {"mean": 3800.0}},
        },
        "k_analysis": {
            "k_range": [2, 3, 4],
            "silhouette_scores": [0.42, 0.30, 0.25],
            "optimal_k": 2,
            "forced_k_min": False,
        },
        "optimal_k": 2,
        "n_clusters": 2,
        "selected_algorithm": "KMeans",
        "selection_reasoning": "KMeans fijo",
        "refinement_reason": "Sin refinar",
        "refinement_count": 1,
        "log_messages": [
            f"[{k}] ok"
            for k in (
                "ingest", "profile", "column_selection", "preprocess",
                "optimize_k", "cluster", "interpret", "refinement",
            )
        ],
    }


class _FakeGraph:
    def __init__(self, final_state, fail_after_first=False):
        self._final = final_state
        self._fail = fail_after_first

    def stream(self, initial_state, stream_mode="values"):
        yield {**initial_state, "log_messages": ["[ingest] ok", "[profile] ok"]}
        if self._fail:
            raise RuntimeError("boom interno")
        yield self._final


def _sse_events(text: str) -> list:
    events = []
    for line in text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: "):]))
    return events


def _saved_record(run_id="abc123abc123") -> dict:
    return build_run_record(
        make_final_state(),
        run_id=run_id,
        created_at="2026-06-10T00:00:00+00:00",
        file_name="t.csv",
        dataset_context="encuesta de prueba",
    )


def _qa_result() -> DataQAResult:
    return DataQAResult(
        narrative="Hay 3 filas en ese grupo.",
        operation="count",
        table=pd.DataFrame({"conteo": [3]}),
    )


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
class TestHealth:
    def test_health_ok(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        assert res.json()["status"] == "ok"


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class TestDatasets:
    def test_upload_csv(self, client):
        res = client.post(
            "/api/datasets/upload",
            files={"file": ("test.csv", CSV_BYTES, "text/csv")},
        )
        assert res.status_code == 200
        payload = res.json()
        assert payload["dataset_id"]
        assert payload["n_rows"] == 12
        assert payload["n_cols"] == 3
        assert payload["donut"]["segments"]
        assert payload["preview"]["columns"] == ["edad", "ingreso", "ciudad"]

    def test_upload_unsupported_extension(self, client):
        res = client.post(
            "/api/datasets/upload",
            files={"file": ("data.xyz", b"whatever", "application/octet-stream")},
        )
        assert res.status_code == 400

    def test_sample(self, client):
        res = client.post("/api/datasets/sample")
        assert res.status_code == 200
        assert res.json()["file_name"] == "customers.csv"

    def test_get_dataset_404(self, client):
        res = client.get("/api/datasets/feedbeefcafe")
        assert res.status_code == 404

    def test_suggest_columns_mocked(self, client, monkeypatch):
        from api.routers import datasets as datasets_router

        def fake_suggest(df, dataset_context=None):
            return {
                "filtered_df": df[["edad", "ingreso"]],
                "static_filter_result": {"kept": ["edad", "ingreso"], "dropped": []},
                "column_recommendation": {"recommended": [{"column": "edad", "importance": "alta"}]},
                "llm_error": None,
            }

        monkeypatch.setattr(datasets_router, "suggest_columns", fake_suggest)
        up = client.post("/api/datasets/upload", files={"file": ("t.csv", CSV_BYTES, "text/csv")})
        dataset_id = up.json()["dataset_id"]

        res = client.post(f"/api/datasets/{dataset_id}/suggest-columns", json={"context": "clientes"})
        assert res.status_code == 200
        payload = res.json()
        assert payload["available_columns"] == ["edad", "ingreso"]
        assert payload["llm_error"] is None

    def test_dataset_chat_mocked(self, client, monkeypatch):
        from api.routers import datasets as datasets_router

        monkeypatch.setattr(datasets_router, "answer_data_question", lambda *a, **kw: _qa_result())
        up = client.post("/api/datasets/upload", files={"file": ("t.csv", CSV_BYTES, "text/csv")})
        dataset_id = up.json()["dataset_id"]

        res = client.post(f"/api/datasets/{dataset_id}/chat", json={"question": "¿cuántas filas hay?"})
        assert res.status_code == 200
        payload = res.json()
        assert "3 filas" in payload["narrative"]
        assert payload["table"]["columns"] == ["conteo"]


# --------------------------------------------------------------------------- #
# Analyze (SSE)
# --------------------------------------------------------------------------- #
class TestAnalyze:
    def _upload(self, client) -> str:
        res = client.post("/api/datasets/upload", files={"file": ("t.csv", CSV_BYTES, "text/csv")})
        return res.json()["dataset_id"]

    def test_analyze_dataset_404(self, client):
        res = client.post("/api/datasets/feedbeefcafe/analyze", json={"context": ""})
        assert res.status_code == 404

    def test_analyze_without_api_key_emits_error_event(self, client, monkeypatch):
        dataset_id = self._upload(client)
        monkeypatch.setattr(settings, "openrouter_api_key", "")
        res = client.post(f"/api/datasets/{dataset_id}/analyze", json={"context": ""})
        events = _sse_events(res.text)
        assert events[-1]["type"] == "error"
        assert events[-1]["error_type"] == "no_api_key"

    def test_analyze_full_flow(self, client, monkeypatch):
        from api.routers import analyze as analyze_router

        monkeypatch.setattr(analyze_router, "compile_graph", lambda: _FakeGraph(make_final_state()))
        dataset_id = self._upload(client)

        res = client.post(f"/api/datasets/{dataset_id}/analyze", json={"context": "encuesta"})
        assert res.status_code == 200
        events = _sse_events(res.text)

        assert events[0]["type"] == "progress"
        assert {s["key"] for s in events[0]["steps"]} >= {"ingest", "cluster", "interpret"}
        done = events[-1]
        assert done["type"] == "done"
        run_id = done["run_id"]
        assert done["summary"]["n_archetypes"] == 2

        # El run quedó persistido y servible.
        rec = client.get(f"/api/runs/{run_id}")
        assert rec.status_code == 200
        assert rec.json()["id"] == run_id

        listing = client.get("/api/runs").json()["runs"]
        assert any(r["id"] == run_id for r in listing)

    def test_analyze_pipeline_error_emits_sse_error(self, client, monkeypatch):
        from api.routers import analyze as analyze_router

        monkeypatch.setattr(
            analyze_router, "compile_graph",
            lambda: _FakeGraph(make_final_state(), fail_after_first=True),
        )
        dataset_id = self._upload(client)
        res = client.post(f"/api/datasets/{dataset_id}/analyze", json={"context": ""})
        events = _sse_events(res.text)
        assert events[-1]["type"] == "error"
        assert events[-1]["failed_step"] is not None


# --------------------------------------------------------------------------- #
# Runs: get / list / chat / exports / delete
# --------------------------------------------------------------------------- #
class TestRuns:
    def test_get_run_404(self, client):
        assert client.get("/api/runs/feedbeefcafe").status_code == 404

    def test_run_id_format_is_enforced(self, client):
        # IDs fuera del formato 12-hex nunca llegan a tocar el filesystem.
        assert store.get_run("../escape") is None
        assert store.delete_run("../../etc/passwd") is False
        assert client.get("/api/runs/not-a-run-id").status_code == 404

    def test_get_and_list_run(self, client):
        store.save_run(_saved_record())
        res = client.get("/api/runs/abc123abc123")
        assert res.status_code == 200
        record = res.json()
        assert record["archetypes"][0]["label"] == "Exploradores prudentes"
        listing = client.get("/api/runs").json()["runs"]
        assert listing[0]["id"] == "abc123abc123"
        assert listing[0]["n_archetypes"] == 2

    def test_exports(self, client):
        store.save_run(_saved_record())

        arch = client.get("/api/runs/abc123abc123/export/archetypes.csv")
        assert arch.status_code == 200
        assert "Exploradores prudentes" in arch.text

        labeled = client.get("/api/runs/abc123abc123/export/labeled.csv")
        assert labeled.status_code == 200
        assert "Arquetipo" in labeled.text

        report = client.get("/api/runs/abc123abc123/export/report.md")
        assert report.status_code == 200
        assert "Exploradores prudentes" in report.text

    def test_run_chat_mocked(self, client, monkeypatch):
        from api.routers import runs as runs_router

        monkeypatch.setattr(runs_router, "answer_data_question", lambda *a, **kw: _qa_result())
        store.save_run(_saved_record())
        res = client.post("/api/runs/abc123abc123/chat", json={"question": "¿cuál es más grande?"})
        assert res.status_code == 200
        assert "3 filas" in res.json()["narrative"]

    def test_delete_run(self, client):
        store.save_run(_saved_record())
        assert client.delete("/api/runs/abc123abc123").json() == {"ok": True}
        assert client.get("/api/runs/abc123abc123").status_code == 404
        assert client.delete("/api/runs/abc123abc123").status_code == 404


# --------------------------------------------------------------------------- #
# Unit: serialization
# --------------------------------------------------------------------------- #
class TestToJsonable:
    def test_numpy_scalars(self):
        assert to_jsonable(np.int64(7)) == 7
        assert to_jsonable(np.float64(1.5)) == 1.5
        assert to_jsonable(np.bool_(True)) is True

    def test_nan_and_inf_become_none(self):
        assert to_jsonable(float("nan")) is None
        assert to_jsonable(np.float64("inf")) is None
        assert to_jsonable(pd.NaT) is None

    def test_non_string_dict_keys_become_str(self):
        assert to_jsonable({0: "a", np.int64(1): "b"}) == {"0": "a", "1": "b"}

    def test_dataframe_and_ndarray(self):
        df = pd.DataFrame({"x": [1, np.nan]})
        assert to_jsonable(df) == [{"x": 1}, {"x": None}]
        assert to_jsonable(np.array([1, 2])) == [1, 2]

    def test_result_is_strict_json(self):
        blob = {"a": np.float64("nan"), "b": [np.int32(1)], 2: pd.Timestamp("2026-06-10")}
        json.dumps(to_jsonable(blob), allow_nan=False)  # no debe lanzar


# --------------------------------------------------------------------------- #
# Unit: build_run_record / run_summary
# --------------------------------------------------------------------------- #
class TestBuildRunRecord:
    def test_record_shape_and_behavioral_fields(self):
        record = _saved_record()
        assert record["optimal_k"] == 2
        assert record["quality"]["score"] == 0.42
        card = record["archetypes"][0]
        for field in (
            "comportamiento_principal", "microcomportamientos", "barreras",
            "habilitadores", "oportunidades_accion", "nivel_cautela",
        ):
            assert field in card
        assert card["size"] == 3
        assert card["prevalence"] == 50.0
        assert record["charts"]["scatter"]
        assert record["charts"]["radar"]["axes"] == ["edad", "ingreso"]
        json.dumps(record, allow_nan=False)  # JSON estricto, sin NaN

    def test_run_summary_defensive_on_partial_record(self):
        summary = run_summary({})  # un archivo viejo/corrupto no debe romper la lista
        assert summary["quality"]["grade"] == "—"

    def test_run_summary_projection(self):
        summary = run_summary(_saved_record())
        assert summary["n_archetypes"] == 2
        assert summary["archetype_labels"] == ["Exploradores prudentes", "Adoptantes tempranos"]
