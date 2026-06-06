"""FastAPI application entry point.

Run locally:  uvicorn api.main:app --reload --port 8000
"""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import analyze, datasets, runs
from src.config.settings import settings

app = FastAPI(
    title="Archetype Suite API",
    description="JSON/SSE API around the LangGraph archetype-clustering pipeline.",
    version="1.0.0",
)

_origins_env = os.environ.get("ARCHETYPE_CORS_ORIGINS", "*")
_origins = ["*"] if _origins_env.strip() == "*" else [o.strip() for o in _origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(datasets.router)
app.include_router(analyze.router)
app.include_router(runs.router)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "has_api_key": bool(settings.openrouter_api_key),
        "models": {
            "selection": settings.llm_model,
            "narrative": settings.llm_narrative_model,
        },
    }


@app.get("/")
def root() -> dict:
    return {"name": "Archetype Suite API", "docs": "/docs", "health": "/api/health"}
