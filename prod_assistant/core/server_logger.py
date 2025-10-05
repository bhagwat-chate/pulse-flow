# prod_assistant/core/server_logger.py

import os
import logging
import structlog
from datetime import datetime
from prod_assistant.core.trace import get_trace_id


def get_mcp_logger(trace_id: str = None):
    """
    Returns a dedicated Structlog logger that writes to a file specific to the trace_id.
    Reuses the Structlog JSON renderer for consistency.
    """
    trace_id = trace_id or get_trace_id()
    date_str = datetime.utcnow().strftime("%Y%m%d")

    log_dir = os.path.join("logs", "mcp_server", date_str)
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"trace_{trace_id}.log")

    # Configure a file handler (non-propagating)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))

    # Create a standard logger (isolated, per-trace)
    base_logger = logging.getLogger(f"mcp_{trace_id}")
    base_logger.handlers.clear()
    base_logger.addHandler(file_handler)
    base_logger.propagate = False
    base_logger.setLevel(logging.INFO)

    # Bind structlog wrapper
    mcp_logger = structlog.wrap_logger(
        base_logger,
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
            structlog.processors.add_log_level,
            structlog.processors.EventRenamer(to="event"),
            structlog.processors.JSONRenderer()
        ],
    )

    return mcp_logger
