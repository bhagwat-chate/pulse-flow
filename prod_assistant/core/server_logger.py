"""
================================================================================
 PulseFlow Server-Level Structured Logging Module
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : core.server_logger
- Version     : 1.0.0
- Created on  : 2025-10-10
- Last Updated: 2025-10-10
- Environment : Python 3.11.13 | StructLog | FastAPI | MCP Adapter
================================================================================

This module provides structured logging utilities specifically for **MCP servers**
within the PulseFlow ecosystem. It complements the global logger by generating
per-trace, file-based JSON logs for asynchronous processes or background workers.

Core Responsibilities
---------------------
- Create trace-specific log files for each MCP execution context.
- Provide lightweight structured logging for non-FastAPI processes.
- Maintain consistency with PulseFlow’s global logging schema.
- Support lifecycle logging during initialization and shutdown.

Workflow Topology
-----------------
    MCP Worker → get_mcp_logger(trace_id) → log_stage(stage, event, level)

External Integrations
---------------------
- **LangSmith** — Enables trace-level log ingestion for workflow observability.
- **AWS CloudWatch** — Reads per-trace log files for system monitoring.
- **prod_assistant.core.trace** — Supplies contextual trace IDs for correlation.
- **prod_assistant.core.globals** — Provides runtime configuration if required.

Changelog
---------
v1.0.0 (2025-10-10)
    • Introduced dedicated MCP structured logging utilities.
    • Added per-trace JSON log files and stage-level lifecycle logging.
    • Ensured safe log creation, platform metadata, and error isolation.

License
-------
Copyright © 2025 Bhagwat Chate.
This code is part of the **PulseFlow** system under the personal projects umbrella.
All rights reserved.
"""

import os
import logging
import structlog
import platform
from datetime import datetime
from prod_assistant.core.trace import get_trace_id


# ---------------------------------------------------------------------
# MCP Structured File Logger
# ---------------------------------------------------------------------
def get_mcp_logger(trace_id: str = None):
    """
    Create or retrieve a structured JSON logger dedicated to a specific trace.

    Parameters
    ----------
    trace_id : str, optional
        The correlation ID for the current execution context.
        If not provided, it is derived from the active context via `get_trace_id()`.

    Returns
    -------
    structlog.BoundLogger
        Structlog logger configured to write to a per-trace file.

    Behavior
    --------
    - Creates a daily log directory under `logs/mcp_server/<YYYYMMDD>/`.
    - Writes JSON-formatted logs per trace (trace_<id>.log).
    - Ensures cross-process trace consistency for async MCP agents.

    Raises
    ------
    OSError
        If log directory or file cannot be created.
    """
    try:
        trace_id = trace_id or get_trace_id()
        date_str = datetime.utcnow().strftime("%Y%m%d")

        log_dir = os.path.join("logs", "mcp_server", date_str)
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(log_dir, f"trace_{trace_id}.log")

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(message)s"))

        base_logger = logging.getLogger(f"mcp_{trace_id}")
        base_logger.handlers.clear()
        base_logger.addHandler(file_handler)
        base_logger.propagate = False
        base_logger.setLevel(logging.INFO)

        mcp_logger = structlog.wrap_logger(
            base_logger,
            processors=[
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                structlog.processors.add_log_level,
                structlog.processors.EventRenamer(to="event"),
                structlog.processors.JSONRenderer(),
            ],
        )
        return mcp_logger

    except OSError as e:
        fallback_logger = logging.getLogger("fallback_mcp_logger")
        fallback_logger.setLevel(logging.WARNING)
        fallback_logger.warning(f"[get_mcp_logger] Failed to create MCP log directory: {e}")
        return structlog.wrap_logger(fallback_logger, processors=[structlog.processors.JSONRenderer()])
    except Exception as e:
        fallback_logger = logging.getLogger("fallback_mcp_logger")
        fallback_logger.setLevel(logging.WARNING)
        fallback_logger.warning(f"[get_mcp_logger] Unexpected error: {e}")
        return structlog.wrap_logger(fallback_logger, processors=[structlog.processors.JSONRenderer()])


# ---------------------------------------------------------------------
# Lifecycle / Stage Logger
# ---------------------------------------------------------------------
def log_stage(stage: str, event: str, level: str = "info", **context):
    """
    Log structured startup or runtime lifecycle events for the MCP server.

    Parameters
    ----------
    stage : str
        Logical stage identifier (e.g., "startup", "retriever_init", "shutdown").
    event : str
        Human-readable event message describing the operation.
    level : str, optional
        Log severity level (default: "info"). Accepts "info", "warning", "error", or "critical".
    **context : dict
        Optional metadata such as environment, process ID, hostname, etc.

    Behavior
    --------
    - Appends standardized metadata: timestamp, trace_id, host, PID, Python version.
    - Writes structured log entries to per-trace log file.
    - Gracefully handles missing trace or logging errors.
    """
    try:
        trace_id = get_trace_id() or "no-trace-id"
        context.update({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "stage": stage,
            "trace_id": trace_id,
            "hostname": platform.node(),
            "pid": os.getpid(),
            "python_version": platform.python_version(),
        })

        logger = get_mcp_logger(trace_id)
        log_func = getattr(logger, level, logger.info)
        log_func(event, **context)

    except Exception as e:
        fallback_logger = logging.getLogger("fallback_stage_logger")
        fallback_logger.setLevel(logging.WARNING)
        fallback_logger.warning(f"[log_stage] Failed to log stage event: {e}")
