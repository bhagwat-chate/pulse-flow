# prod_assistant/retriever/retrieval.py
"""
================================================================================
 PulseFlow – Retriever Module
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : retriever.retrieval
- Version     : 1.0.0
- Created on  : 2025-10-10
- Last Updated: 2025-10-10
- Environment : Python 3.11.13 | LangChain | AstraDB | StructLog
================================================================================

Purpose
--------
Provides a centralized, environment-aware retrieval layer for the PulseFlow
system. It integrates **AstraDB** vector search with **LangChain’s contextual
compression retriever** to deliver high-precision recall for product-related
queries.

Core Responsibilities
---------------------
1. Load AstraDB vector store using model and keyspace configuration.
2. Use embeddings + LLM models provided by `ModelLoader`.
3. Apply contextual compression via `LLMChainFilter` for relevance filtering.
4. Expose clean retrieval interfaces to agents and MCP servers.

Workflow Topology
-----------------
    AgenticRAG → Retriever → AstraDBVectorStore → ContextualCompressionRetriever

External Integrations
---------------------
- **AstraDB** – Vector database for semantic recall.
- **LangChain** – Provides retriever abstraction and compressor classes.
- **ModelLoader** – Supplies embeddings and LLM models.
- **StructLog** – Structured event logging with trace_id propagation.

Changelog
---------
v1.0.0 (2025-10-10)
    • Added contextual compression retriever integration.
    • Implemented structured exception handling with ProductAssistantException.
    • Added robust initialization checks for AstraDB config.
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
    """
    Central retriever class that handles AstraDB vector retrieval and
    contextual compression filtering.

    Behavior
    --------
    - Initializes from environment-specific configuration (dev/prod).
    - Loads LLM and embedding models via ModelLoader.
    - Provides compressed semantic retrieval with `ContextualCompressionRetriever`.
    """

    def __init__(self):
        try:
            self.config = get_config()
            self.model_loader = ModelLoader()

            astra_cfg = self.config.get("astra", {})
            self.api_endpoint = astra_cfg.get("api_endpoint")
            self.keyspace = astra_cfg.get("keyspace")
            self.token = astra_cfg.get("token")
            self.collection_name = astra_cfg.get("collection_name", "pulseflow_collection")

            retriever_cfg = self.config.get("retriever", {"top_k": 3})
            self.top_k = retriever_cfg.get("top_k", 3)

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
    def _load_vector_store(self):
        """
        Create and initialize the AstraDBVectorStore.

        Behavior
        --------
        - Loads the embedding model using `ModelLoader`.
        - Connects to AstraDB using endpoint, token, and keyspace.
        - Raises `ProductAssistantException` on any connection or schema error.

        Returns
        -------
        AstraDBVectorStore
            The instantiated AstraDB vector store client.
        """
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
    def load_retriever(self):
        """
        Build and return the contextual retriever instance.

        Behavior
        --------
        - Ensures the AstraDB vector store is loaded first.
        - Wraps base retriever with `ContextualCompressionRetriever` for
          post-retrieval filtering using LLM reasoning.
        - Caches retriever instance for reuse across calls.

        Returns
        -------
        ContextualCompressionRetriever
            A fully configured LangChain retriever ready for use.
        """
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
    def call_retriever(self, query: str):
        """
        Execute a semantic query against the AstraDB retriever.

        Behavior
        --------
        - Automatically loads or reuses retriever instance.
        - Performs contextual retrieval with compression filtering.
        - Logs query and document count for observability.
        - Raises `ProductAssistantException` if retrieval fails.

        Parameters
        ----------
        query : str
            User or agent query text to search in the vector store.

        Returns
        -------
        List[Document]
            List of LangChain `Document` objects matching the query.
        """
        try:
            retriever = self.load_retriever()
            LOGGER.info("Invoking retriever", query=query)

            results = retriever.invoke(query)
            LOGGER.info("Retriever results fetched", count=len(results))
            return results

        except Exception as e:
            LOGGER.error("Retriever query failed", query=query, error=str(e))
            raise ProductAssistantException("Retriever query failed", e)
