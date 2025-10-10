# prod_assistant/router/exception_handlers.py
"""
Exception Handlers
==================
Defines global error handlers for FastAPI routes,
ensuring structured, consistent JSON error responses.
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from prod_assistant.core.globals import LOGGER
from prod_assistant.exception.custom_exception import ProductAssistantException


def register_exception_handlers(app):
    """Attach global exception handlers to the FastAPI app."""

    @app.exception_handler(ProductAssistantException)
    async def product_assistant_exception_handler(request: Request, exc: ProductAssistantException):
        LOGGER.error("ProductAssistantException caught globally", error=str(exc))
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": exc.error_message,
                "file": exc.file_name,
                "line": exc.lineno,
                "trace_id": request.headers.get("X-Trace-ID", "unknown"),
            },
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        wrapped = ProductAssistantException(exc)
        LOGGER.critical("Unhandled system exception", error=str(wrapped))
        return JSONResponse(
            status_code=500,
            content={
                "error": "Unexpected Error",
                "message": "An unexpected error occurred.",
                "trace_id": request.headers.get("X-Trace-ID", "unknown"),
            },
        )
