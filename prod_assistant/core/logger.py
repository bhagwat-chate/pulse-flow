"""
================================================================================
 PulseFlow Structured Logging Module
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : core.logger
- Version     : 1.0.0
- Created on  : 2025-10-07
- Last Updated: 2025-10-10
- Environment : Python 3.11.13 | StructLog | FastAPI | CloudWatch
================================================================================

This module implements the **CustomLogger** class used for structured,
JSON-formatted logging across the entire PulseFlow system. It integrates
Python's standard logging with `structlog` to produce consistent, traceable
logs compatible with cloud observability tools such as AWS CloudWatch and
LangSmith.

Core Responsibilities
---------------------
- Initialize structured log files with timestamp-based filenames.
- Configure both console and file handlers for runtime visibility.
- Inject the `trace_id` from the active request context into every log record.
- Ensure JSON serialization for easy ingestion by observability systems.

Workflow Topology
-----------------
bootstrap_app() → CustomLogger().get_logger() → globals.LOGGER → all modules

External Integrations
---------------------
- **StructLog** — Provides JSON rendering and contextual enrichment.
- **FastAPI Middleware** — Assigns per-request `trace_id` context.
- **LangSmith** — Consumes structured logs for trace-level analysis.
- **AWS CloudWatch** — Reads rotating log files for centralized monitoring.

Changelog
---------
v1.0.0 (2025-10-07)
    • Initial structured logging implementation using StructLog.
    • Added per-request trace injection via `trace_id`.
    • Integrated console + file handlers for unified observability.

License
-------
Copyright © 2025 Bhagwat Chate.
This code is part of the **PulseFlow** system under the personal projects umbrella.
All rights reserved.
"""

import os
import logging
from datetime import datetime
import structlog
from prod_assistant.core.trace import get_trace_id


class CustomLogger:
    """
    Provides a unified structured logging interface using StructLog.

    This class is responsible for configuring application-wide logging
    during bootstrap. It ensures both file-based and console-based
    logging are JSON-rendered, timestamped, and enriched with `trace_id`.

    Attributes
    ----------
    logs_dir : str
        Directory path where log files will be stored.
    log_file_path : str
        Full path of the active log file being written.
    """

    def __init__(self, log_dir: str = "logs") -> None:
        """
        Initialize a structured logger with log directory setup.

        Parameters
        ----------
        log_dir : str, optional
            The directory where log files are stored (default: 'logs').

        Raises
        ------
        OSError
            If the log directory cannot be created due to permission issues.
        """
        try:
            self.logs_dir = os.path.join(os.getcwd(), log_dir)
            os.makedirs(self.logs_dir, exist_ok=True)

            log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
            self.log_file_path = os.path.join(self.logs_dir, log_file)
        except OSError as e:
            raise OSError(f"Failed to initialize log directory: {e}")

    # ------------------------------------------------------------------
    # StructLog Processor
    # ------------------------------------------------------------------
    def add_trace_id(self, logger, method_name, event_dict):
        """
        Structlog processor that injects the current `trace_id` into every log entry.

        Parameters
        ----------
        logger : Any
            The logger instance emitting the log.
        method_name : str
            Name of the logging method being called (e.g., 'info', 'error').
        event_dict : dict
            The event data being logged.

        Returns
        -------
        dict
            The updated event dictionary containing the active `trace_id`.
        """
        try:
            event_dict["trace_id"] = get_trace_id()
        except Exception:
            event_dict["trace_id"] = "no-trace-id"
        return event_dict

    # ------------------------------------------------------------------
    # Logger Factory
    # ------------------------------------------------------------------
    def get_logger(self, name: str = __file__):
        """
        Create and configure a structured logger instance.

        Parameters
        ----------
        name : str, optional
            The module or file name associated with the logger.

        Returns
        -------
        structlog.BoundLogger
            Configured JSON-based StructLog logger instance.

        Notes
        -----
        - Writes structured logs to both console and file outputs.
        - Ensures consistency between human-readable and machine-parsed logs.
        - Automatically adds ISO UTC timestamps and log levels.
        """
        try:
            logger_name = os.path.basename(name)

            # File handler setup
            file_handler = logging.FileHandler(self.log_file_path, mode="a", encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(logging.Formatter("%(message)s"))

            # Console handler setup
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(logging.Formatter("%(message)s"))

            # Root logging configuration
            logging.basicConfig(
                level=logging.INFO,
                format="%(message)s",
                handlers=[console_handler, file_handler],
                force=True,
            )

            # StructLog configuration
            structlog.configure(
                processors=[
                    structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                    structlog.processors.add_log_level,
                    structlog.processors.EventRenamer(to="event"),
                    self.add_trace_id,
                    structlog.processors.JSONRenderer()
                ],
                context_class=dict,
                wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
                cache_logger_on_first_use=True,
            )

            return structlog.get_logger(logger_name)

        except Exception as e:
            # Minimal safe fallback logger
            fallback_logger = logging.getLogger("fallback_logger")
            fallback_logger.setLevel(logging.INFO)
            fallback_logger.warning(f"Failed to initialize StructLog logger: {e}")
            return fallback_logger
