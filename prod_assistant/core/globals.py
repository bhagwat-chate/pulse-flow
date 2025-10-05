"""
Central registry for app-wide singletons like config, logger, and cache.
Ensures they are initialized once and reused everywhere.
"""

from typing import Optional, Any
import contextvars

# ----------------------------------------------------------------------
# Global singletons
# ----------------------------------------------------------------------
CONFIG: Optional[Any] = None
LOGGER: Optional[Any] = None
CACHE: Optional[Any] = None

# ----------------------------------------------------------------------
# Global trace context (shared across processes)
# ----------------------------------------------------------------------
TRACE_ID: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="no-trace-id")


def set_config(cfg):
    """Called once at bootstrap to register config globally."""
    global CONFIG
    CONFIG = cfg


def get_config():
    """Safely return loaded config."""
    if CONFIG is None:
        raise RuntimeError("Config accessed before initialization. Call bootstrap_app() first.")
    return CONFIG
