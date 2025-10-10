"""
================================================================================
 PulseFlow Trace Context Management Module
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : core.trace
- Version     : 1.0.0
- Created on  : 2025-10-07
- Last Updated: 2025-10-10
- Environment : Python 3.11.13 | FastAPI | LangGraph | StructLog
================================================================================

This module manages the **distributed trace context** across all layers of
the PulseFlow system. It ensures that every request, background task, and
agentic workflow shares a consistent `trace_id`, enabling end-to-end
observability through logs, metrics, and LangSmith traces.

Core Responsibilities
---------------------
- Generate unique per-request trace identifiers (UUIDv4).
- Store and retrieve the active trace context using ContextVars.
- Support automatic propagation of `trace_id` across FastAPI requests,
  background threads, and async LangGraph processes.
- Provide safe fallbacks when trace context is missing.

Workflow Topology
-----------------
FastAPI Middleware → new_trace_id() → TRACE_ID.set() → get_trace_id() → Logger

External Integrations
---------------------
- **core.globals** — Stores the global ContextVar TRACE_ID.
- **core.logger** — Injects trace_id into structured logs automatically.
- **LangSmith** — Associates workflow traces with the same trace_id.
- **CloudWatch / ELK** — Enables end-to-end trace correlation.

Changelog
---------
v1.0.0 (2025-10-10)
    • Standardized UUIDv4 trace ID generation and retrieval.
    • Ensured safe context propagation across async boundaries.
    • Added error handling and fallback for missing trace states.

License
-------
Copyright © 2025 Bhagwat Chate.
This code is part of the **PulseFlow** system under the personal projects umbrella.
All rights reserved.
"""

import uuid
from prod_assistant.core.globals import TRACE_ID


# ---------------------------------------------------------------------
# Trace Context Utilities
# ---------------------------------------------------------------------
def new_trace_id() -> str:
    """
    Generate and register a new unique trace ID in the global context.

    Returns
    -------
    str
        Newly generated UUIDv4 trace identifier string.

    Behavior
    --------
    - Creates a new trace_id using the `uuid` module.
    - Sets it into the shared ContextVar (`TRACE_ID`) so that all
      subsequent operations within the current request or process
      automatically inherit it.

    Example
    -------
    # >>> trace_id = new_trace_id()
    # >>> print(trace_id)
    '23b5a3c8-f4e0-42a8-98b2-1b2a1e90ad22'
    """
    try:
        trace_id = str(uuid.uuid4())
        TRACE_ID.set(trace_id)
        return trace_id
    except Exception:
        # Safe fallback in case of context variable errors
        return "no-trace-id"


def get_trace_id() -> str:
    """
    Retrieve the current active trace ID from the context.

    Returns
    -------
    str
        The currently active trace identifier if available, else `"no-trace-id"`.

    Behavior
    --------
    - Fetches the active trace_id from the ContextVar `TRACE_ID`.
    - Provides a default `"no-trace-id"` when unset.

    Example
    -------
    >>> get_trace_id()
    '23b5a3c8-f4e0-42a8-98b2-1b2a1e90ad22'
    """
    try:
        return TRACE_ID.get() or "no-trace-id"
    except Exception:
        return "no-trace-id"
