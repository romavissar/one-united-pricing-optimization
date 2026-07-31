"""Map domain exceptions to HTTP responses. Bad input is never a 500."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.exceptions import IdentificationError, InfeasibleModelError, SchemaError


def install_exception_handlers(app: FastAPI) -> None:
    """Register handlers so domain failures become structured 4xx bodies."""

    @app.exception_handler(SchemaError)
    async def _schema(_request: Request, exc: SchemaError) -> JSONResponse:
        status = 403 if str(exc).startswith("REFUSED") else 400
        return JSONResponse(
            status_code=status,
            content={"error": "schema_error", "detail": str(exc)},
        )

    @app.exception_handler(IdentificationError)
    async def _identification(
        _request: Request, exc: IdentificationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "identification_error", "detail": str(exc)},
        )

    @app.exception_handler(InfeasibleModelError)
    async def _infeasible(
        _request: Request, exc: InfeasibleModelError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": "infeasible", "detail": str(exc)},
        )


def error_body(code: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"error": code, "detail": detail, **extra}
