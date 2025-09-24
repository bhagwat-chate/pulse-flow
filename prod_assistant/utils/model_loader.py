# prod_assistant/utils/model_loader.py

import os
import sys
import json
import asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from prod_assistant.utils.config_loader import load_config
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from prod_assistant.utils.api_key_manager import ApiKeyManager
from dotenv import load_dotenv
from prod_assistant.logger import GLOBAL_LOGGER as log
from prod_assistant.exception.custom_exception import ProductAssistantException


class ModelLoader:
    def __init__(self):
        if os.getenv("ENV", "local").lower() != "production":
            load_dotenv()
            log.info("run mode: LOCAL : .env load successful")
        else:
            log.info("run mode: PRODUCTION")

        self.api_key_mgr = ApiKeyManager()
        self.config = load_config()

        log.info("config load complete", config_keys=list(self.config.keys()))

    def load_embeddings(self):
        try:
            model_name = self.config["embedding_model"]['model_name']
            log.info("loading embedding model", model=model_name)

            # patch: ensure an event loop exists for gRPC aio
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())

            return GoogleGenerativeAIEmbeddings(
                model=model_name,
                google_api_key=self.api_key_mgr.get("GOOGLE_API_KEY")
            )

        except ProductAssistantException as e:
            log.error("Error in embedding model load", error=str(e))
            raise ProductAssistantException("failed to load embedding model", sys)

    def load_llm(self):
        try:
            llm_block = self.config['llm']
            provider_key = os.getenv('LLM_PROVIDER', 'openai')

            if provider_key not in llm_block:
                log.error('invalid LLM provider name', provider=provider_key)
                raise ValueError('invalid LLM provider name')

            llm_config = llm_block[provider_key]
            provider = llm_config.get('provider')
            model_name = llm_config.get('model_name')
            temperature = llm_config.get('temperature', 0.2)
            max_tokens = llm_config.get('max_output_tokens', 2048)

            log.info("loading LLM...", provider=provider, model=model_name)

            if provider == 'google':
                return ChatGoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    google_api_key=self.api_key_mgr.api_keys.get("GOOGLE_API_KEY")
                )

            elif provider == 'groq':
                return ChatGroq(
                    model=model_name,
                    api_key=self.api_key_mgr.api_keys.get('GROQ_API_KEY'),
                    temperature=temperature
                )

            elif provider == 'openai':
                return ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    api_key=self.api_key_mgr.api_keys.get("OPENAI_API_KEY")
                )

        except ProductAssistantException as e:
            log.error("Error in LLM load", error=str(e))
            raise ProductAssistantException("failed to load LLM load", sys)