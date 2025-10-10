# prod_assistant/utils/ragas_helper.py
"""
================================================================================
 RAGAs Evaluation Helper
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : utils.ragas_helper
- Version     : 1.0.0
- Created on  : 2025-10-10
- Environment : Python 3.11.13 | LangChain | RAGAs | LangSmith | StructLog
================================================================================

Description
-----------
Provides asynchronous utilities for executing and tracking RAGAs evaluation
metrics (Context Precision & Response Relevancy) and forwarding them to
LangSmith for unified observability.

Architecture Context
--------------------
Layer:        Evaluation / Metrics
Upstream:     AgenticRAG Workflow
Downstream:   LangSmith + RAGAs Metrics Dashboards

Key Responsibilities
--------------------
• Compute Context Precision and Response Relevancy scores asynchronously.
• Safely integrate results into LangSmith traces using `attach_ragas_metrics()`.
• Provide a non-blocking, event-loop-safe evaluation launcher for production use.

Engineering Standards
---------------------
• FAANGM-grade async-safe design and logging.
• Behavior-style docstrings across all functions.
• Trace-aware semantic logging (`trace_id`, `context_precision`, `response_relevancy`).
• Never blocks or raises in production mode — all failures are logged, not thrown.
"""

import asyncio
from prod_assistant.core.globals import LOGGER
from prod_assistant.core.trace import get_trace_id
from prod_assistant.evaluation.ragas_eval import (
    evaluate_context_precision,
    evaluate_response_relevancy,
)
from prod_assistant.utils.langsmith_utils import attach_ragas_metrics


# ======================================================================
# Internal Async RAGAs Evaluation Executor
# ======================================================================
async def _run_ragas_async(question: str, response: str, retrieved_contexts):
    """
    Execute both RAGAs evaluation metrics asynchronously.

    Behavior
    --------
    - Computes two independent scores:
        1. Context Precision (semantic alignment of retrieved context)
        2. Response Relevancy (faithfulness of final answer)
    - Logs metrics under the current `trace_id`.
    - Attaches scores to LangSmith (if supported).

    Parameters
    ----------
    question : str
        Original user query passed to the agentic system.
    response : str
        Final generated answer from the LLM pipeline.
    retrieved_contexts : list or str
        List of retrieved document contexts or a single string.

    Returns
    -------
    None
        Metrics are logged and attached asynchronously.

    Raises
    ------
    None
        All exceptions are caught internally and logged.
    """
    trace_id = get_trace_id()
    try:
        ctx_precision = await evaluate_context_precision(question, response, retrieved_contexts)
        resp_relevancy = await evaluate_response_relevancy(question, response, retrieved_contexts)

        LOGGER.info(
            "RAGAs evaluation completed successfully",
            trace_id=trace_id,
            context_precision=ctx_precision,
            response_relevancy=resp_relevancy,
        )

        # Attach metrics to LangSmith trace for observability
        attach_ragas_metrics(ctx_precision, resp_relevancy)

    except Exception as e:
        LOGGER.warning(
            "RAGAs evaluation failed or skipped",
            trace_id=trace_id,
            error=str(e),
        )


# ======================================================================
# Public Safe Launcher
# ======================================================================
def launch_ragas_evaluation(question: str, response: str, docs):
    """
    Safely launch RAGAs evaluation regardless of async loop context.

    Behavior
    --------
    - Detects if an asyncio event loop is already running.
    - If running → schedules `_run_ragas_async()` as a background task.
    - If not running → creates a new loop and runs it synchronously.
    - Logs start and failure messages with `trace_id`.
    - Never blocks main execution or raises exceptions.

    Parameters
    ----------
    question : str
        Original input query used during agent processing.
    response : str
        Generated output from the LLM.
    docs : list or str
        Retrieved document(s) used for context evaluation.

    Returns
    -------
    None
        Executes asynchronously or synchronously based on runtime context.

    Raises
    ------
    None
        All exceptions are caught and logged without propagation.
    """
    trace_id = get_trace_id()
    try:
        retrieved_contexts = [docs] if isinstance(docs, str) else docs

        try:
            # Use existing loop if running inside async environment
            loop = asyncio.get_running_loop()
            loop.create_task(_run_ragas_async(question, response, retrieved_contexts))
            LOGGER.info("RAGAs async evaluation scheduled", trace_id=trace_id)
        except RuntimeError:
            # For sync contexts: create and run a temporary event loop
            asyncio.run(_run_ragas_async(question, response, retrieved_contexts))
            LOGGER.info("RAGAs evaluation executed in new event loop", trace_id=trace_id)

    except Exception as e:
        LOGGER.warning(
            "Failed to start RAGAs evaluation block",
            trace_id=trace_id,
            error=str(e),
        )
