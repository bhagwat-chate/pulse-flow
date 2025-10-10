# prod_assistant/utils/model_loader.py
"""
================================================================================
 Model Loader Utility
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : utils.model_loader
- Version     : 1.0.0
- Created on  : 2025-10-10
- Environment : Python 3.11.13 | LangChain | OpenAI | Groq | Gemini
================================================================================

Description
-----------
Centralized factory responsible for dynamically loading embedding and LLM clients
based on the unified YAML configuration (config_base.yaml + environment overrides).

Architecture Context
--------------------
Layer:        Core Utilities
Upstream:     AgenticRAG, Retriever, RAGAs Evaluator
Downstream:   OpenAI, Groq, Google Gemini API Providers

Key Responsibilities
--------------------
• Load embedding models from multiple supported providers.
• Load LLMs for generative and reasoning workflows.
• Provide configuration-driven model initialization with structured logging.
• Enforce error safety through standardized `ProductAssistantException` handling.

Supported Providers
-------------------
| Provider | Embedding | Chat LLM |
|-----------|------------|-----------|
| OpenAI    | ✅ Yes     | ✅ Yes    |
| Groq      | ✅ (via OpenAI API schema) | ✅ Yes |
| Google    | ✅ text-embedding-004 | ✅ Gemini-Pro |

Engineering Standards
---------------------
• Follows FAANGM-grade structured logging with trace correlation.
• Ensures consistent failure wrapping for observability and debugging.
• Behavior-oriented docstrings across all methods.
• Config-driven instantiation (no hardcoded model keys).
"""

import sys
from prod_assistant.core.globals import get_config, LOGGER
from prod_assistant.exception.custom_exception import ProductAssistantException

# LangChain integrations
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


class ModelLoader:
    """
    Dynamically creates Embedding and LLM clients using configuration data.

    Behavior
    --------
    - Reads unified configuration (`config.yaml` or environment overrides).
    - Supports provider-based branching for OpenAI, Groq, and Google.
    - Logs all model loading activities and failures.
    - Wraps exceptions using `ProductAssistantException` for consistent error flow.
    """

    def __init__(self):
        """
        Initialize the ModelLoader and load runtime configuration.

        Behavior
        --------
        - Retrieves configuration from the global registry via `get_config()`.
        - Logs application context for traceability.
        """
        try:
            self.config = get_config()
            LOGGER.info(
                "ModelLoader initialized successfully",
                app=self.config["app"]["name"],
                env=self.config["app"].get("env", "base"),
            )
        except Exception as e:
            LOGGER.error("Failed to initialize ModelLoader", error=str(e))
            raise ProductAssistantException("ModelLoader initialization failed", e)

    # ------------------------------------------------------------------
    # Embedding Loader
    # ------------------------------------------------------------------
    def load_embeddings(self):
        """
        Initialize and return an embedding model client.

        Behavior
        --------
        - Selects the embedding provider and model from configuration.
        - Supports OpenAI, Groq, and Google Generative AI.
        - Ensures uniform initialization and structured logging.
        - Raises `ProductAssistantException` for unsupported providers.

        Returns
        -------
        object
            Initialized embedding model client (LangChain-compatible).

        Raises
        ------
        ProductAssistantException
            If initialization fails or provider is unsupported.
        """
        try:
            provider = self.config["embedding"]["provider"]
            model = self.config["embedding"]["model"]
            api_key = self.config["embedding"].get("api_key")

            LOGGER.info("Loading embedding model", provider=provider, model=model)

            if provider == "openai":
                return OpenAIEmbeddings(model=model, openai_api_key=api_key)

            elif provider == "groq":
                # Groq mirrors OpenAI’s embedding API schema
                return OpenAIEmbeddings(model=model, openai_api_key=api_key)

            elif provider == "google":
                return GoogleGenerativeAIEmbeddings(model=model, google_api_key=api_key)

            else:
                LOGGER.error("Unsupported embedding provider", provider=provider)
                raise ProductAssistantException(
                    f"Unsupported embedding provider: {provider}", sys
                )

        except Exception as e:
            LOGGER.error("Failed to initialize embedding model", error=str(e))
            raise ProductAssistantException("Embedding model initialization failed", e)

    # ------------------------------------------------------------------
    # LLM Loader
    # ------------------------------------------------------------------
    def load_llm(self):
        """
        Initialize and return a Chat LLM model client.

        Behavior
        --------
        - Selects provider and model details from configuration.
        - Supports OpenAI, Groq, and Google Gemini models.
        - Applies sensible defaults for temperature and token limits.
        - Handles both synchronous and async LLM APIs uniformly.

        Returns
        -------
        object
            Initialized LangChain LLM client instance.

        Raises
        ------
        ProductAssistantException
            If initialization fails or provider is unsupported.
        """
        try:
            provider = self.config["llm"]["provider"]
            model = self.config["llm"]["model"]
            api_key = self.config["llm"].get("api_key")
            temperature = self.config["llm"].get("temperature", 0.2)
            max_tokens = self.config["llm"].get("max_output_tokens", 2048)

            LOGGER.info("Loading LLM model", provider=provider, model=model)

            if provider == "openai":
                return ChatOpenAI(
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

            elif provider == "groq":
                return ChatGroq(
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                )

            elif provider == "google":
                return ChatGoogleGenerativeAI(
                    model=model,
                    google_api_key=api_key,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )

            else:
                LOGGER.error("Unsupported LLM provider", provider=provider)
                raise ProductAssistantException(
                    f"Unsupported LLM provider: {provider}", sys
                )

        except Exception as e:
            LOGGER.error("Failed to initialize LLM", error=str(e))
            raise ProductAssistantException("LLM initialization failed", e)
