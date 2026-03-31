import pandas as pd

from src.data.profiler import DataProfiler
from src.models.state import PipelineState


def profile_node(state: PipelineState) -> dict:
    df = pd.DataFrame(state["raw_data"])
    profiler = DataProfiler()
    profile = profiler.profile(df)

    return {
        "profile": profile,
        "log_messages": [
            f"[profile] Profiled {profile['n_cols']} columns: "
            f"{len(profile['numeric_columns'])} numeric, "
            f"{len(profile['categorical_columns'])} categorical"
        ],
    }
