# prod_assistant/router/trace_middleware.py
"""
Trace Middleware
================
Attaches a unique `trace_id` to every HTTP request and response.
"""

from fastapi import Request
from prod_assistant.core.trace import new_trace_id
from prod_assistant.core.globals import LOGGER


def add_trace_middleware(app):
    """Register middleware that injects trace_id into request/response."""

    @app.middleware("http")
    async def add_trace_id(request: Request, call_next):
        trace_id = new_trace_id()
        LOGGER.info("New request trace initialized", trace_id=trace_id, path=request.url.path)
        response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id
        return response
