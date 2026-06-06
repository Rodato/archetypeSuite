"""Session + run storage.

- Datasets live in memory (uploaded → analyzed in the same session; ephemeral is fine for
  the "SaaS look, no DB" scope the project chose).
- Completed runs are persisted as JSON files on disk so "Mis análisis" survives restarts —
  a real local history without standing up a database.
"""
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

DATA_DIR = Path(os.environ.get("ARCHETYPE_DATA_DIR", Path(__file__).resolve().parent / "_data"))
RUNS_DIR = DATA_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex[:12]


# --------------------------------------------------------------------------- #
# In-memory dataset store
# --------------------------------------------------------------------------- #
class _DatasetStore:
    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def add(self, df: pd.DataFrame, file_name: str) -> str:
        dataset_id = new_id()
        with self._lock:
            self._data[dataset_id] = {
                "df": df,
                "file_name": file_name,
                "created_at": now_iso(),
            }
        return dataset_id

    def get(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._data.get(dataset_id)

    def get_df(self, dataset_id: str) -> Optional[pd.DataFrame]:
        entry = self.get(dataset_id)
        return entry["df"] if entry else None


datasets = _DatasetStore()


# --------------------------------------------------------------------------- #
# On-disk run store
# --------------------------------------------------------------------------- #
def save_run(record: Dict[str, Any]) -> None:
    path = RUNS_DIR / f"{record['id']}.json"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    path = RUNS_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def list_runs() -> List[Dict[str, Any]]:
    runs = []
    for path in RUNS_DIR.glob("*.json"):
        try:
            runs.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return runs


def delete_run(run_id: str) -> bool:
    path = RUNS_DIR / f"{run_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False
