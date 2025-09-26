import asyncio
import grpc.experimental.aio as grpc_aio
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas.metrics import ResponseRelevancy
from prod_assistant.utils.model_loader import ModelLoader

grpc_aio.init_grpc_aio()
model_loader = ModelLoader()


def evaluate_context_precision():
    """
    _summary_
    :return:
    """
    pass


def evaluate_response_relevancy():
    """
    Args:
    query (_type_): _description_
    :return:
    """
    pass
