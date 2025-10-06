# prod_assistant/utils/langsmith_utils.py
from langsmith import run_helpers
from prod_assistant.core.globals import LOGGER
from prod_assistant.core.trace import get_trace_id


def attach_ragas_metrics(ctx_precision: float, resp_relevancy: float):
    """Adds RAGAs metrics to LangSmith run (if supported)."""
    trace_id = get_trace_id()
    try:
        if hasattr(run_helpers, "add_extra"):
            run_helpers.add_extra({
                "ragas_context_precision": ctx_precision,
                "ragas_response_relevancy": resp_relevancy,
            })
            LOGGER.info("RAGAs metrics attached to LangSmith", trace_id=trace_id)
        else:
            LOGGER.warning(
                "Failed to attach RAGAs metrics to LangSmith",
                trace_id=trace_id,
                error="Unsupported version (no add_extra)",
            )
    except Exception as e:
        LOGGER.warning("LangSmith metric attach failed", trace_id=trace_id, error=str(e))
