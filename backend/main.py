"""FastAPI application entrypoint.

Phase 0: health. Phase 6: full route table in `src.api.routes`.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.errors import install_exception_handlers
from src.api.routes import router

APP_VERSION = "0.1.0"

app = FastAPI(title="Pricing & Release Optimizer", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

install_exception_handlers(app)
app.include_router(router)


@app.get("/api/health")
def health() -> dict[str, str]:
    """Liveness probe. Returns status and app version."""
    return {"status": "ok", "version": APP_VERSION}
