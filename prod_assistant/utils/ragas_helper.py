# prod_assistant/utils/ragas_helper.py
import asyncio
from prod_assistant.core.globals import LOGGER
from prod_assistant.core.trace import get_trace_id
from prod_assistant.evaluation.ragas_eval import (
    evaluate_context_precision,
    evaluate_response_relevancy,
)
from prod_assistant.utils.langsmith_utils import attach_ragas_metrics


async def _run_ragas_async(question, response, retrieved_contexts):
    """Executes both RAGAs metrics asynchronously."""
    trace_id = get_trace_id()
    try:
        ctx_precision = await evaluate_context_precision(question, response, retrieved_contexts)
        resp_relevancy = await evaluate_response_relevancy(question, response, retrieved_contexts)

        LOGGER.info(
            "RAGAs evaluation complete",
            trace_id=trace_id,
            context_precision=ctx_precision,
            response_relevancy=resp_relevancy,
        )

        attach_ragas_metrics(ctx_precision, resp_relevancy)
    except Exception as e:
        LOGGER.warning("RAGAs evaluation skipped or failed", trace_id=trace_id, error=str(e))


def launch_ragas_evaluation(question, response, docs):
    """Safe wrapper to launch RAGAs eval, works in and outside async loops."""
    trace_id = get_trace_id()
    try:
        retrieved_contexts = [docs] if isinstance(docs, str) else docs
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_run_ragas_async(question, response, retrieved_contexts))
        except RuntimeError:
            asyncio.run(_run_ragas_async(question, response, retrieved_contexts))
    except Exception as e:
        LOGGER.warning("RAGAs evaluation block failed to start", trace_id=trace_id, error=str(e))
