# prod_assistant/evaluation/ragas_eval.py

"""
================================================================================
 PulseFlow Evaluation – RAGAS Metric Computation Module
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : evaluation.ragas_eval
- Version     : 1.0.0
- Created on  : 2025-10-10
- Last Updated: 2025-10-10
- Environment : Python 3.11.13 | LangChain | RAGAS | AsyncIO | gRPC
================================================================================

This module implements asynchronous metric evaluation using **RAGAS**
to assess the performance of PulseFlow’s Retrieval-Augmented Generation (RAG)
responses. It measures two primary metrics:

1. **Context Precision** – How precisely the retrieved context supports the answer.
2. **Response Relevancy** – How relevant and semantically aligned the final answer is.

Core Responsibilities
---------------------
- Wrap RAGAS metric evaluators into async LangChain-compatible pipelines.
- Dynamically load model and embedding instances via `ModelLoader`.
- Compute per-sample scores for single-turn RAG interactions.
- Provide coroutine-safe metric accessors used by `ragas_helper.py`.

Workflow Topology
-----------------
    AgenticRAG → RAGAsHelper.launch_ragas_evaluation()
    → ragas_eval.evaluate_context_precision()
    → ragas_eval.evaluate_response_relevancy()

External Integrations
---------------------
- **RAGAS** — Provides evaluation metrics.
- **LangchainLLMWrapper / EmbeddingsWrapper** — Interfaces RAGAS with LangChain models.
- **ModelLoader** — Loads OpenAI/Groq/Gemini models as configured.
- **LangSmith / CloudWatch** — Receives metric traces for observability.

Changelog
---------
v1.0.0 (2025-10-10)
    • Initial implementation of async RAGAS evaluation.
    • Added context precision and response relevancy scorers.
    • Integrated LangChain model loader and gRPC async initialization.

License
-------
Copyright © 2025 Bhagwat Chate.
This code is part of the **PulseFlow** system under the personal projects umbrella.
All rights reserved.
"""

from prod_assistant.utils.model_loader import ModelLoader
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference, ResponseRelevancy
import grpc.experimental.aio as grpc_aio
from prod_assistant.core.globals import LOGGER


# ---------------------------------------------------------------------
# Initialize async gRPC runtime
# ---------------------------------------------------------------------
try:
    grpc_aio.init_grpc_aio()
except Exception as e:
    if LOGGER:
        LOGGER.warning("gRPC async initialization failed", error=str(e))


# ---------------------------------------------------------------------
# Async evaluation: Context Precision
# ---------------------------------------------------------------------
async def _evaluate_context_precision_async(query: str, response: str, retrieved_context: list) -> float:
    """
    Evaluate the contextual precision score for a given RAG sample.

    Parameters
    ----------
    query : str
        Original user query or prompt.
    response : str
        Generated answer from the model.
    retrieved_context : list
        List of retrieved documents or context strings.

    Returns
    -------
    float
        Context precision score (0.0–1.0 range).
    """
    try:
        model_loader = ModelLoader()
        llm = model_loader.load_llm()
        evaluator_llm = LangchainLLMWrapper(llm)
        metric = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=retrieved_context,
        )
        score = await metric.single_turn_ascore(sample)
        LOGGER.info("Context precision evaluated", score=score)
        return score
    except Exception as e:
        LOGGER.error("Context precision evaluation failed", error=str(e))
        return 0.0


# ---------------------------------------------------------------------
# Async evaluation: Response Relevancy
# ---------------------------------------------------------------------
async def _evaluate_response_relevancy_async(query: str, response: str, retrieved_context: list) -> float:
    """
    Evaluate the semantic relevancy of a model’s response to its retrieved context.

    Parameters
    ----------
    query : str
        User query prompting the model.
    response : str
        Model-generated answer text.
    retrieved_context : list
        List of contextual passages used during retrieval.

    Returns
    -------
    float
        Response relevancy score (0.0–1.0 range).
    """
    try:
        model_loader = ModelLoader()
        llm = model_loader.load_llm()
        evaluator_llm = LangchainLLMWrapper(llm)
        embedding_model = model_loader.load_embeddings()
        evaluator_embeddings = LangchainEmbeddingsWrapper(embedding_model)
        scorer = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)

        sample = SingleTurnSample(
            user_input=query,
            response=response,
            retrieved_contexts=retrieved_context,
        )
        score = await scorer.single_turn_ascore(sample)
        LOGGER.info("Response relevancy evaluated", score=score)
        return score
    except Exception as e:
        LOGGER.error("Response relevancy evaluation failed", error=str(e))
        return 0.0


# ---------------------------------------------------------------------
# Public Async Wrappers
# ---------------------------------------------------------------------
async def evaluate_context_precision(query: str, response: str, retrieved_context: list) -> float:
    """
    Public async entrypoint for context precision metric.

    Returns
    -------
    float
        The evaluated context precision score.
    """
    return await _evaluate_context_precision_async(query, response, retrieved_context)


async def evaluate_response_relevancy(query: str, response: str, retrieved_context: list) -> float:
    """
    Public async entrypoint for response relevancy metric.

    Returns
    -------
    float
        The evaluated response relevancy score.
    """
    return await _evaluate_response_relevancy_async(query, response, retrieved_context)
