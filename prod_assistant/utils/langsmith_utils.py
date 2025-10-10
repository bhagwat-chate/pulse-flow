# prod_assistant/utils/langsmith_utils.py
"""
================================================================================
 LangSmith Utilities
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : utils.langsmith_utils
- Version     : 1.0.0
- Created on  : 2025-10-10
- Environment : Python 3.11.13 | LangSmith | RAGAs | StructLog
================================================================================

Description
-----------
Provides helper functions for safely attaching RAGAs evaluation metrics to
LangSmith runs for unified observability and trace correlation.

Architecture Context
--------------------
Layer:        Evaluation / Observability
Upstream:     AgenticRAG Workflow
Downstream:   LangSmith Experiments + RAGAs Dashboards

Key Responsibilities
--------------------
• Attach context precision and response relevancy metrics to LangSmith trace.
• Maintain backward compatibility with multiple LangSmith SDK versions.
• Log outcomes with consistent trace_id correlation.

Engineering Standards
---------------------
• Follows FAANGM-grade docstring structure and semantic logging.
• Operates in a best-effort mode (never breaks workflow if LangSmith unsupported).
• Fully compatible with async or sync execution contexts.
"""

from langsmith import run_helpers
from prod_assistant.core.globals import LOGGER
from prod_assistant.core.trace import get_trace_id


def attach_ragas_metrics(ctx_precision: float, resp_relevancy: float):
    """
    Attach RAGAs evaluation metrics to the active LangSmith run.

    Behavior
    --------
    - Fetches the current `trace_id` from the global trace context.
    - Uses `run_helpers.add_extra()` (if available) to attach
      `ragas_context_precision` and `ragas_response_relevancy` fields.
    - Logs success or gracefully warns if the SDK version lacks support.

    Parameters
    ----------
    ctx_precision : float
        Context Precision score computed by RAGAs (0–1 range).
    resp_relevancy : float
        Response Relevancy score computed by RAGAs (0–1 range).

    Returns
    -------
    None
        Operates in best-effort mode. Metrics are added to LangSmith metadata
        if supported; otherwise only warning logs are emitted.

    Raises
    ------
    None
        This function intentionally suppresses all exceptions to ensure
        non-blocking telemetry integration.
    """
    trace_id = get_trace_id()
    try:
        if hasattr(run_helpers, "add_extra"):
            run_helpers.add_extra({
                "ragas_context_precision": ctx_precision,
                "ragas_response_relevancy": resp_relevancy,
            })
            LOGGER.info(
                "RAGAs metrics successfully attached to LangSmith",
                trace_id=trace_id,
                context_precision=ctx_precision,
                response_relevancy=resp_relevancy,
            )
        else:
            LOGGER.warning(
                "LangSmith SDK does not support metric attachment (no add_extra)",
                trace_id=trace_id,
            )
    except Exception as e:
        LOGGER.warning(
            "Failed to attach RAGAs metrics to LangSmith",
            trace_id=trace_id,
            error=str(e),
        )
