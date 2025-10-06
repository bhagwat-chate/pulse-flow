# prod_assistant/evaluation/ragas_eval.py

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
