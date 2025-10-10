# prod_assistant/exception/custom_exception.py

"""
================================================================================
 PulseFlow Exception Handling Module
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : exception.custom_exception
- Version     : 1.0.0
- Created on  : 2025-10-10
- Last Updated: 2025-10-10
- Environment : Python 3.11.13 | FastAPI | StructLog | LangGraph
================================================================================

This module defines a unified **ProductAssistantException** class that provides
a standardized, structured approach for error handling across the entire
PulseFlow system. It enhances Python's base `Exception` by embedding file name,
line number, and traceback details into a concise, logger-friendly format.

Core Responsibilities
---------------------
- Capture and normalize exception context from multiple sources (e.g., `sys`, `Exception`).
- Extract the deepest traceback frame to report the most relevant error location.
- Provide structured string representations for logging and JSON serialization.
- Maintain full traceback for observability and debugging while staying safe for production.

Workflow Topology
-----------------
    try → raise ProductAssistantException(e) → logged → FastAPI JSONResponse

External Integrations
---------------------
- **core.logger** — Consumes exceptions in structured log events.
- **router.main** — Uses this class for global exception handling in API routes.
- **AWS CloudWatch / LangSmith** — Correlates errors via trace_id and structured fields.

Changelog
---------
v1.0.0 (2025-10-10)
    • Introduced unified exception class with contextual traceback.
    • Standardized message structure for all service layers.
    • Added compatibility with `sys.exc_info()` and direct exception wrapping.

License
-------
Copyright © 2025 Bhagwat Chate.
This code is part of the **PulseFlow** system under the personal projects umbrella.
All rights reserved.
"""

import sys
import traceback
from typing import Optional, cast


class ProductAssistantException(Exception):
    """
    Standardized custom exception used throughout the PulseFlow system.

    Automatically extracts file name, line number, and traceback from the
    raised exception for structured observability.

    Parameters
    ----------
    error_message : str | Exception
        Descriptive message or raw exception object to be wrapped.
    error_details : Optional[object], optional
        Optional source of exception details (e.g., `sys`, `Exception`, or None).

    Attributes
    ----------
    file_name : str
        Source file in which the exception occurred.
    lineno : int
        Line number of the error location.
    error_message : str
        Normalized exception message.
    traceback_str : str
        Full formatted traceback string, if available.
    """

    def __init__(self, error_message, error_details: Optional[object] = None):
        try:
            # Normalize message string
            if isinstance(error_message, BaseException):
                norm_msg = str(error_message)
            else:
                norm_msg = str(error_message)

            # Resolve exc_info depending on type of details
            exc_type = exc_value = exc_tb = None
            if error_details is None:
                exc_type, exc_value, exc_tb = sys.exc_info()
            else:
                if hasattr(error_details, "exc_info"):
                    exc_info_obj = cast(sys, error_details)
                    exc_type, exc_value, exc_tb = exc_info_obj.exc_info()
                elif isinstance(error_details, BaseException):
                    exc_type, exc_value, exc_tb = (
                        type(error_details),
                        error_details,
                        error_details.__traceback__,
                    )
                else:
                    exc_type, exc_value, exc_tb = sys.exc_info()

            # Walk to the last traceback frame
            last_tb = exc_tb
            while last_tb and last_tb.tb_next:
                last_tb = last_tb.tb_next

            self.file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
            self.lineno = last_tb.tb_lineno if last_tb else -1
            self.error_message = norm_msg

            # Full traceback string (optional)
            if exc_type and exc_tb:
                self.traceback_str = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            else:
                self.traceback_str = ""

            super().__init__(self.__str__())

        except Exception as internal_error:
            # Safe fallback if error construction itself fails
            self.file_name = "<unknown>"
            self.lineno = -1
            self.error_message = f"Failed to initialize exception: {internal_error}"
            self.traceback_str = ""
            super().__init__(self.error_message)

    # ------------------------------------------------------------------
    def __str__(self) -> str:
        """
        Return a concise, logger-friendly error string.

        Returns
        -------
        str
            Compact error summary including file, line, and message.
        """
        base = f"Error in [{self.file_name}] at line [{self.lineno}] | Message: {self.error_message}"
        if self.traceback_str:
            return f"{base}\nTraceback:\n{self.traceback_str}"
        return base

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        """
        Developer-friendly representation of the exception object.

        Returns
        -------
        str
            Reconstructable object-style representation.
        """
        return (
            f"ProductAssistantException("
            f"file={self.file_name!r}, line={self.lineno}, message={self.error_message!r})"
        )
