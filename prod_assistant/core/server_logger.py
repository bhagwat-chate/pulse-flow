# # prod_assistant/core/server_logger.py

import os
import sys
import json
import socket
import logging
import structlog
import platform
from datetime import datetime
from prod_assistant.core.trace import get_trace_id
from prod_assistant.core.globals import get_config


# ==========================================================
# 🔹 MCP Structured File Logger
# ==========================================================
def get_mcp_logger(trace_id: str = None):
    """
    Returns a dedicated Structlog logger that writes to a per-trace log file.
    Consistent with global structured logging format for cross-system observability.

    Args:
        trace_id (str, optional): Correlation ID for this trace context.

    Returns:
        structlog.BoundLogger: Logger bound to the given trace_id.
    """
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


# ==========================================================
# 🔹 Stage / Lifecycle Logger (used for startup + diagnostics)
# ==========================================================
def log_stage(stage: str, event: str, level: str = "info", **context):
    """
    Log structured startup or runtime information for the MCP server.

    Args:
        stage (str): Logical stage name (e.g., "startup", "retriever_init").
        event (str): Human-readable event description.
        level (str): Log level ("info", "warning", "error", "critical").
        **context: Arbitrary metadata (env, pid, hostname, etc.)
    """
    trace_id = get_trace_id() or "no-trace-id"
    context.update({
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "stage": stage,
        "trace_id": trace_id,
        "hostname": platform.node(),
        "pid": os.getpid(),
        "python_version": platform.python_version(),
    })

    # Create or reuse the per-trace logger
    logger = get_mcp_logger(trace_id)

    # Emit structured log at requested level
    log_func = getattr(logger, level, logger.info)
    log_func(event, **context)