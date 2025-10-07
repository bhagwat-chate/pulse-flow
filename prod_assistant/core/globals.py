# prod_assistant/core/globals.py

"""
================================================================================
 PulseFlow Global Registry Module
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : core.globals
- Version     : 1.0.0
- Created on  : 2025-10-07
- Last Updated: 2025-10-07
- Environment : Python 3.11.13 | FastAPI | LangGraph | AWS RDS | StructLog
================================================================================

This module defines the **centralized global registry** used across the entire
PulseFlow system to manage and share single-instance runtime objects such as
configuration, logger, cache, and contextual trace identifiers.

Core Responsibilities
---------------------
- Maintain globally accessible singletons (`CONFIG`, `LOGGER`, `CACHE`).
- Provide safe getter/setter interfaces for application-wide config access.
- Expose `TRACE_ID` context variable for distributed request tracing.
- Ensure thread-safe, process-level persistence of global runtime state.

Workflow Topology
-----------------
    bootstrap_app() → globals.set_config() → globals.get_config() → used by all modules

External Integrations
---------------------
- **bootstrap.py** — Initializes and registers config + logger into globals.
- **logger.py** — Supplies the structured logger instance (StructLog).
- **FastAPI / LangGraph** — Accesses global config for workflow orchestration.
- **AWS Secrets Manager** — Injects environment-specific configuration values.

Changelog
---------
v1.0.0  (2025-10-07)
    • Introduced unified global registry for configuration, logger, and cache.
    • Added ContextVar-based TRACE_ID for multi-thread trace propagation.
    • Standardized safe initialization and retrieval methods for CONFIG object.

License
-------
Copyright © 2025 Bhagwat Chate.
This code is part of the **PulseFlow** system under the personal projects umbrella.
All rights reserved.
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
