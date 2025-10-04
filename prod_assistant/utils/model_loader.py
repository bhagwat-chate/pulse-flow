# prod_assistant/utils/model_loader.py
"""
Model Loader Module
===================

Centralized factory for loading Embedding and LLM clients
based on unified YAML configuration (config_base.yaml + env overrides).

Supports:
---------
- OpenAI  → Embeddings + Chat LLM
- Groq    → Embeddings + Chat LLM
- Google  → Embeddings (text-embedding-004) + Chat LLM (Gemini)

Author: Bhagwat Chate
Project: PulseFlow – E-commerce Product Intelligence
Version: 1.0.0
"""

import sys
from prod_assistant.core.globals import get_config, LOGGER
from prod_assistant.exception.custom_exception import ProductAssistantException

# LangChain integrations
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


class ModelLoader:
    """Dynamically creates Embedding and LLM clients from the unified config."""

    def __init__(self):
        self.config = get_config()
        LOGGER.info("ModelLoader initialized", app=self.config["app"]["name"], env=self.config["app"].get("env", "base"))

    # ------------------------------------------------------------------
    # Embedding Loader
    # ------------------------------------------------------------------
    def load_embeddings(self):
        """Return initialized embedding model client."""
        try:
            provider = self.config["embedding"]["provider"]
            model = self.config["embedding"]["model"]
            api_key = self.config["embedding"].get("api_key")

            LOGGER.info("🔹 Loading Embedding Model", provider=provider, model=model)

            if provider == "openai":
                return OpenAIEmbeddings(model=model, openai_api_key=api_key)

            elif provider == "groq":
                # Groq currently mirrors OpenAI embedding schema
                return OpenAIEmbeddings(model=model, openai_api_key=api_key)

            elif provider == "google":
                return GoogleGenerativeAIEmbeddings(
                    model=model,
                    google_api_key=api_key
                )

            else:
                LOGGER.error("Unsupported embedding provider", provider=provider)
                raise ProductAssistantException(f"Unsupported embedding provider: {provider}", sys)

        except Exception as e:
            LOGGER.error("Failed to initialize embedding model", error=str(e))
            raise ProductAssistantException("Embedding model initialization failed", sys)

    # ------------------------------------------------------------------
    # LLM Loader
    # ------------------------------------------------------------------
    def load_llm(self):
        """Return initialized LLM model client."""
        try:
            provider = self.config["llm"]["provider"]
            model = self.config["llm"]["model"]
            api_key = self.config["llm"].get("api_key")
            temperature = self.config["llm"].get("temperature", 0.2)
            max_tokens = self.config["llm"].get("max_output_tokens", 2048)

            LOGGER.info("Loading LLM", provider=provider, model=model)

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
                raise ProductAssistantException(f"Unsupported LLM provider: {provider}", sys)

        except Exception as e:
            LOGGER.error("Failed to initialize LLM", error=str(e))
            raise ProductAssistantException("LLM initialization failed", sys)
