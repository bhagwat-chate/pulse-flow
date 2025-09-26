# prod_assistant/retriever/retrieval.py

import os
from langchain_astradb import AstraDBVectorStore
from typing import List
from langchain_core.documents import Document
from prod_assistant.utils.model_loader import ModelLoader
from dotenv import load_dotenv
from prod_assistant.utils.config_loader import load_config

from langchain.retrievers.document_compressors import LLMChainFilter
from langchain.retrievers import ContextualCompressionRetriever
from prod_assistant.evaluation.ragas_eval import evaluate_context_precision
from prod_assistant.evaluation.ragas_eval import evaluate_response_relevancy


class Retriever:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.config = load_config()
        self._load_env_variables()
        self.vstore = None
        self.retriever = None

    def _load_env_variables(self):
        load_dotenv()

        required_vars = ['OPENAI_API_KEY', 'GROQ_API_KEY', 'GOOGLE_API_KEY', 'ASTRA_DB_ENDPOINT',
                         'ASTRA_DB_APPLICATION_TOKEN', 'ASTRA_DB_KEYSPACE']

        missing_vars = [var for var in required_vars if os.getenv(var) is None]

        if missing_vars:
            raise EnvironmentError(f"missing environment variables: {missing_vars}")

        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.db_api_endpoint = os.getenv('ASTRA_DB_ENDPOINT')
        self.db_application_token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
        self.db_keyspace = os.getenv('ASTRA_DB_KEYSPACE')

    def load_retriever(self):
        """
        Initialize and return a ContextualCompressionRetriever
        wrapping an AstraDB Vector Store retriever with MMR search.

        Returns:
            ContextualCompressionRetriever: A retriever instance for semantic search with compression.
        """
        if not self.vstore:
            collection_name = self.config['astra_db']['collection_name']
            self.vstore = AstraDBVectorStore(
                embedding=self.model_loader.load_embeddings(),
                collection_name=collection_name,
                api_endpoint=self.db_api_endpoint,
                token=self.db_application_token,
                namespace=self.db_keyspace,
            )

        if not self.retriever:
            # --- Hard-coded MMR params ---
            mmr_retriever = self.vstore.as_retriever(
                search_type='mmr',
                search_kwargs={
                    "k": 5,  # final top_k docs
                    "lambda_mult": 0.5,  # relevance vs diversity tradeoff
                    "fetch_k": 20,  # initial candidate pool
                    "score_threshold": 0.0  # min similarity filter
                }
            )

            llm = self.model_loader.load_llm()
            compressor = LLMChainFilter.from_llm(llm)

            self.retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=mmr_retriever
            )

        return self.retriever

    def call_retriever(self, user_query: str):
        """
        Run a query against the retriever and return matching documents.

        Args:
            user_query (str): The natural language query.

        Returns:
            List[Document]: List of relevant documents retrieved.
        """
        retriever = self.load_retriever()
        return retriever.get_relevant_documents(user_query)


if __name__ == '__main__':

    user_query = "Can you suggest good budget iPhone under 1,00,000 INR?"

    retriever_obj = Retriever()
    retrieved_docs = retriever_obj.call_retriever(user_query)

    # for idx, doc in enumerate(results, 1):
    #     print(f"Result {idx}: {doc.page_content}\nMetadata: {doc.metadata}\n")

    formatted_chunks = []

    def _format_docs(docs) -> str:

        if not docs:
            return "No relevant documents found"

        for d in docs:
            meta = d.metadata or {}

            formatted = (
                f"Title: {meta.get('product_title', 'N/A')}\n"
                f"Price: {meta.get('price', 'N/A')}\n"
                f"Rating: {meta.get('rating', 'N/A')}\n"
                f"Reviews: \n{meta.get(d.page_content.strip(), 'N/A')}"
            )

            formatted_chunks.append(formatted)

        return "\n\n---\n\n".join(formatted_chunks)

    retrieved_contexts = [_format_docs(doc) for doc in retrieved_docs]

    # below response is set only for testing purpose.
    response = "iPhone 16 plus, iPhone 15, iPhone 16 are the best iPhone under 1,00,000, INR"
    context_score = evaluate_context_precision(user_query, response, retrieved_contexts)
    relevancy_score = evaluate_response_relevancy(user_query, response, retrieved_contexts)

    print("\n---Evaluation Metrics ---")
    print(f"Context precision score: {context_score}")
    print(f"Response relevancy score: {relevancy_score}")
