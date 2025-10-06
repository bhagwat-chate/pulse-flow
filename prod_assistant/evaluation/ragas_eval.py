# # prod_assistant/evaluation/ragas_eval.py
#
# """
# RAGAs Evaluation Module
# =======================
#
# Purpose:
# --------
# Provides on-demand single-turn evaluation for PulseFlow’s RAG responses using RAGAs metrics.
# Currently supports:
#     • Context Precision (LLM-based, without ground truth)
#     • Response Relevancy (embedding + LLM hybrid metric)
#
# This module integrates with PulseFlow’s unified logger and trace_id propagation,
# allowing correlation between evaluation results and main workflow traces.
#
# Author  : Bhagwat Chate
# Project : PulseFlow – E-commerce Product Intelligence
# Version : 1.1.0 (Evaluation Integration)
# """
#
# import asyncio
# import grpc.experimental.aio as grpc_aio
# from prod_assistant.utils.model_loader import ModelLoader
# from prod_assistant.exception.custom_exception import ProductAssistantException
# from prod_assistant.core.globals import LOGGER
# from prod_assistant.core.trace import get_trace_id
# from ragas import SingleTurnSample
# from ragas.llms import LangchainLLMWrapper
# from ragas.embeddings import LangchainEmbeddingsWrapper
# from ragas.metrics import LLMContextPrecisionWithoutReference, ResponseRelevancy
#
#
# # ----------------------------------------------------------------------
# # Global Initialization
# # ----------------------------------------------------------------------
# grpc_aio.init_grpc_aio()
# model_loader = ModelLoader()
#
#
# # ----------------------------------------------------------------------
# # Metric 1: Context Precision
# # ----------------------------------------------------------------------
# def evaluate_context_precision(query: str, response: str, retrieved_context: list) -> float:
#     """
#     Evaluate the LLM’s ability to use retrieved context faithfully.
#
#     Args:
#         query (str): User question.
#         response (str): Generated model response.
#         retrieved_context (list): List of retrieved context strings.
#
#     Returns:
#         float: Context precision score (0–1).
#     """
#     trace_id = get_trace_id()
#     try:
#         sample = SingleTurnSample(
#             user_input=query,
#             response=response,
#             retrieved_contexts=retrieved_context,
#         )
#
#         async def _evaluate():
#             llm = model_loader.load_llm()
#             evaluator_llm = LangchainLLMWrapper(llm)
#             metric = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
#             score = await metric.single_turn_ascore(sample)
#             return score
#
#         score = asyncio.run(_evaluate())
#         LOGGER.info(
#             "RAGAs Context Precision evaluated",
#             trace_id=trace_id,
#             query=query,
#             score=score,
#         )
#         return score
#
#     except Exception as e:
#         LOGGER.error("RAGAs context precision evaluation failed", error=str(e), trace_id=trace_id)
#         raise ProductAssistantException("Context precision evaluation failed", e)
#
#
# # ----------------------------------------------------------------------
# # Metric 2: Response Relevancy
# # ----------------------------------------------------------------------
# def evaluate_response_relevancy(query: str, response: str, retrieved_context: list) -> float:
#     """
#     Evaluate how relevant the generated response is to the retrieved context.
#
#     Args:
#         query (str): User question.
#         response (str): Model-generated response.
#         retrieved_context (list): List of retrieved context strings.
#
#     Returns:
#         float: Response relevancy score (0–1).
#     """
#     trace_id = get_trace_id()
#     try:
#         sample = SingleTurnSample(
#             user_input=query,
#             response=response,
#             retrieved_contexts=retrieved_context,
#         )
#
#         async def _evaluate():
#             llm = model_loader.load_llm()
#             evaluator_llm = LangchainLLMWrapper(llm)
#             embed_model = model_loader.load_embeddings()
#             evaluator_embeddings = LangchainEmbeddingsWrapper(embed_model)
#             scorer = ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
#             score = await scorer.single_turn_ascore(sample)
#             return score
#
#         score = asyncio.run(_evaluate())
#         LOGGER.info(
#             "RAGAs Response Relevancy evaluated",
#             trace_id=trace_id,
#             query=query,
#             score=score,
#         )
#         return score
#
#     except Exception as e:
#         LOGGER.error("RAGAs response relevancy evaluation failed", error=str(e), trace_id=trace_id)
#         raise ProductAssistantException("Response relevancy evaluation failed", e)


import asyncio
from prod_assistant.utils.model_loader import ModelLoader
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference, ResponseRelevancy
import grpc.experimental.aio as grpc_aio

grpc_aio.init_grpc_aio()
model_loader = ModelLoader()


async def _evaluate_context_precision_async(query, response, retrieved_context):
    llm = model_loader.load_llm()
    evaluator_llm = LangchainLLMWrapper(llm)
    metric = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

    sample = SingleTurnSample(
        user_input=query,
        response=response,
        retrieved_contexts=retrieved_context,
    )
    return await metric.single_turn_ascore(sample)


async def _evaluate_response_relevancy_async(query, response, retrieved_context):
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
    return await scorer.single_turn_ascore(sample)


async def evaluate_context_precision(query, response, retrieved_context):
    """Async wrapper that returns evaluation score."""
    return await _evaluate_context_precision_async(query, response, retrieved_context)


async def evaluate_response_relevancy(query, response, retrieved_context):
    """Async wrapper that returns evaluation score."""
    return await _evaluate_response_relevancy_async(query, response, retrieved_context)
