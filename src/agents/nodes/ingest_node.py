import pandas as pd

from src.data.ingest import DataIngestor
from src.models.state import PipelineState


def ingest_node(state: PipelineState) -> dict:
    ingestor = DataIngestor()

    raw = state["raw_data"]
    df = pd.DataFrame(raw)

    ingestor.validate(df)

    return {
        "raw_data": df.to_dict(orient="list"),
        "original_columns": list(df.columns),
        "log_messages": [
            f"[ingest] Loaded data: {df.shape[0]} rows, {df.shape[1]} columns"
        ],
    }
