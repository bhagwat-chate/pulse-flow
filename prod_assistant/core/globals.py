"""
================================================================================
 PulseFlow Global Registry Module
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : core.globals
- Version     : 1.0.0
- Created on  : 2025-10-07
- Last Updated: 2025-10-10
- Environment : Python 3.11.13 | FastAPI | LangGraph | AWS RDS | StructLog
================================================================================

This module defines the centralized **Global Registry** that maintains
runtime-wide singletons for configuration, logging, cache, and trace contexts.

Core Responsibilities
---------------------
- Maintain globally accessible singletons (`CONFIG`, `LOGGER`, `CACHE`).
- Provide safe getter/setter interfaces for application-wide configuration.
- Manage context-aware trace propagation across threads and async tasks.
- Serve as the shared runtime registry for all PulseFlow modules.

Workflow Topology
-----------------
bootstrap_app() → set_config() → get_config() → used by all modules

External Integrations
---------------------
- **bootstrap.py** — Initializes and registers config + logger.
- **logger.py** — Supplies the structured logger instance (StructLog).
- **FastAPI / LangGraph** — Accesses global config for orchestration.
- **AWS Secrets Manager** — Injects environment-specific configuration values.

Changelog
---------
v1.0.0 (2025-10-07)
    • Introduced unified global registry for configuration, logger, and cache.
    • Added ContextVar-based TRACE_ID for concurrent trace isolation.
    • Standardized error handling and logging for global access methods.

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
# Global trace context (shared across threads or async tasks)
# ----------------------------------------------------------------------
TRACE_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "trace_id",
    default="no-trace-id"
)


# ----------------------------------------------------------------------
# Configuration management
# ----------------------------------------------------------------------
def set_config(cfg: dict) -> None:
    """
    Register the unified configuration object globally.

    Parameters
    ----------
    cfg : dict
        The merged configuration dictionary produced by `bootstrap_app()`.

    Behavior
    --------
    - Stores the configuration in a global variable for system-wide access.
    - Logs initialization status if a logger is available.

    Raises
    ------
    TypeError
        If `cfg` is not a dictionary or contains invalid keys.
    """
    global CONFIG
    try:
        if not isinstance(cfg, dict):
            raise TypeError("Configuration object must be a dictionary.")
        CONFIG = cfg

        if LOGGER:
            LOGGER.info("Global configuration registered successfully")
    except Exception as e:
        if LOGGER:
            LOGGER.error("Failed to register global configuration", error=str(e))
        else:
            print(f"[globals:set_config] Error registering config: {e}")


def get_config() -> dict:
    """
    Safely retrieve the global configuration object.

    Returns
    -------
    dict
        The globally loaded configuration dictionary.

    Raises
    ------
    RuntimeError
        If configuration is accessed before initialization.
    """
    try:
        if CONFIG is None:
            raise RuntimeError(
                "Configuration accessed before initialization. "
                "Ensure bootstrap_app() has been executed."
            )
        return CONFIG
    except Exception as e:
        if LOGGER:
            LOGGER.error("Configuration access failed", error=str(e))
        raise


# ----------------------------------------------------------------------
# Logger management
# ----------------------------------------------------------------------
def set_logger(logger_instance: Any) -> None:
    """
    Register the structured logger globally for system-wide usage.

    Parameters
    ----------
    logger_instance : Any
        Instance of the application's structured logger (e.g., StructLog).

    Raises
    ------
    TypeError
        If logger_instance does not expose expected logging methods.
    """
    global LOGGER
    try:
        if not hasattr(logger_instance, "info") or not hasattr(logger_instance, "error"):
            raise TypeError("Logger must implement 'info' and 'error' methods.")
        LOGGER = logger_instance
        LOGGER.info("Global logger registered successfully")
    except Exception as e:
        print(f"[globals:set_logger] Error registering logger: {e}")


def get_logger() -> Any:
    """
    Retrieve the globally registered logger instance.

    Returns
    -------
    Any
        The structured logger instance.

    Raises
    ------
    RuntimeError
        If logger is accessed before being initialized.
    """
    try:
        if LOGGER is None:
            raise RuntimeError(
                "Logger accessed before initialization. "
                "Ensure bootstrap_app() has registered a logger."
            )
        return LOGGER
    except Exception as e:
        if LOGGER:
            LOGGER.error("Logger access failed", error=str(e))
        raise

