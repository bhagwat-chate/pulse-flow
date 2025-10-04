# prod_assistant/retriever/retrieval.py
"""
Retriever Module
================

Purpose
--------
Centralized retrieval layer for PulseFlow — integrates AstraDB vector search
with LangChain's contextual compression retriever and unified configuration.

Responsibilities
----------------
1. Load AstraDB vector store using environment-aware config.
2. Use embedding + LLM models from ModelLoader.
3. Apply contextual compression for precise retrieval.
4. Provide a clean, reusable retriever interface for agents and MCP servers.

Author  : Bhagwat Chate
Project : PulseFlow – E-commerce Product Intelligence
Version : 1.0.0
"""

from prod_assistant.core.bootstrap import bootstrap_app
from prod_assistant.core.globals import CONFIG, LOGGER, get_config
from prod_assistant.utils.model_loader import ModelLoader
from prod_assistant.exception.custom_exception import ProductAssistantException

from langchain_astradb import AstraDBVectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter


# ----------------------------------------------------------------------
# Bootstrap Application Context
# ----------------------------------------------------------------------
if CONFIG is None:
    bootstrap_app()


class Retriever:
    """Loads and manages AstraDB retriever with contextual compression."""

    def __init__(self):
        try:
            # --- Load configuration ---
            self.config = get_config()
            self.model_loader = ModelLoader()

            astra_cfg = self.config.get("astra", {})
            self.api_endpoint = astra_cfg.get("api_endpoint")
            self.keyspace = astra_cfg.get("keyspace")
            self.token = astra_cfg.get("token")
            self.collection_name = astra_cfg.get("collection_name", "pulseflow_collection")

            retriever_cfg = self.config.get("retriever", {"top_k": 3})
            self.top_k = retriever_cfg.get("top_k", 3)

            # --- Lazy initialization placeholders ---
            self.vstore = None
            self.retriever_instance = None

            LOGGER.info(
                "Retriever initialized",
                api_endpoint=self.api_endpoint,
                keyspace=self.keyspace,
                top_k=self.top_k,
            )

        except Exception as e:
            LOGGER.error("Failed to initialize Retriever", error=str(e))
            raise ProductAssistantException("Retriever initialization failed", e)

    # ------------------------------------------------------------------
    # Build AstraDB Vector Store
    # ------------------------------------------------------------------
    def _load_vector_store(self):
        """Create AstraDBVectorStore with embeddings."""
        try:
            embed_model = self.model_loader.load_embeddings()

            self.vstore = AstraDBVectorStore(
                embedding=embed_model,
                collection_name=self.collection_name,
                api_endpoint=self.api_endpoint,
                token=self.token,
                namespace=self.keyspace,
            )

            LOGGER.info("AstraDB vector store loaded successfully")
            return self.vstore

        except Exception as e:
            LOGGER.error("Failed to load AstraDB vector store", error=str(e))
            raise ProductAssistantException("Vector store load failed", e)

    # ------------------------------------------------------------------
    # Build Contextual Retriever
    # ------------------------------------------------------------------
    def load_retriever(self):
        """Initialize and return a ContextualCompressionRetriever."""
        try:
            if not self.vstore:
                self._load_vector_store()

            if not self.retriever_instance:
                base_retriever = self.vstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={
                        "k": self.top_k,
                        "fetch_k": 20,
                        "lambda_mult": 0.0,
                        "score_threshold": 0.0,
                    },
                )

                llm = self.model_loader.load_llm()
                compressor = LLMChainFilter.from_llm(llm)

                self.retriever_instance = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=base_retriever,
                )

                LOGGER.info("Contextual retriever created successfully", top_k=self.top_k)

            return self.retriever_instance

        except Exception as e:
            LOGGER.error("Failed to initialize retriever instance", error=str(e))
            raise ProductAssistantException("Retriever instance initialization failed", e)

    # ------------------------------------------------------------------
    # Query Interface
    # ------------------------------------------------------------------
    def call_retriever(self, query: str):
        """Run query against AstraDB retriever and return LangChain Document list."""
        try:
            retriever = self.load_retriever()
            LOGGER.info("Invoking retriever", query=query)

            results = retriever.invoke(query)

            LOGGER.info("Retriever results fetched", count=len(results))
            return results

        except Exception as e:
            LOGGER.error("Retriever query failed", query=query, error=str(e))
            raise ProductAssistantException("Retriever query failed", e)


# ----------------------------------------------------------------------
# Debug Entry (Standalone Mode)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    retriever = Retriever()
    query = "What do users say about iPhone 15 Plus battery?"
    docs = retriever.call_retriever(query)
    print(f"Retrieved {len(docs)} documents.")
