"""FastAPI layer that exposes the existing LangGraph archetype pipeline as a JSON/SSE API.

The Python clustering pipeline in `src/` is reused verbatim — this package only adds
HTTP transport, JSON-safe serialization and an in-memory/on-disk session+run store so a
Next.js frontend can drive the same flow the Streamlit UI does.
"""
