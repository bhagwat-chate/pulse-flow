# prod_assistant/core/trace.py

"""
Trace Context Management
========================

Provides utilities to create, fetch, and manage per-request trace IDs.
These IDs propagate across both FastAPI and MCP layers.
"""

import uuid
from prod_assistant.core.globals import TRACE_ID


def new_trace_id() -> str:
    """Generate and set a new unique trace_id in the context."""
    trace_id = str(uuid.uuid4())
    TRACE_ID.set(trace_id)
    return trace_id


def get_trace_id() -> str:
    """Retrieve the active trace_id, or 'no-trace-id' if none is set."""
    return TRACE_ID.get() or "no-trace-id"
