import os
import sys
import json
import asyncio
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from prod_assistant.utils.config_loader import load_config
from langchain_google_genai import ChatGoogleGenerativeAI
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
