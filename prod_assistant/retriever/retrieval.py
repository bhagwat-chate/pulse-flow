# prod_assistant/retriever/retrieval.py

import os
from langchain_astradb import AstraDBVectorStore
from typing import List
from langchain_core.documents import Document
from prod_assistant.utils.model_loader import ModelLoader
from dotenv import load_dotenv
from prod_assistant.utils.config_loader import load_config


class Retriever:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.config = load_config()
        self._load_env_variables()
        self.vstore = None
        self.retriever = None

    def _load_env_variables(self):
        load_dotenv()

        required_vars = ['OPENAI_API_KEY', 'GROQ_API_KEY', 'GOOGLE_API_KEY', 'ASTRA_DB_ENDPOINT', 'ASTRA_DB_APPLICATION_TOKEN', 'ASTRA_DB_KEYSPACE']

        missing_vars = [var for var in required_vars if os.getenv(var) is None]

        if missing_vars:
            raise EnvironmentError(f"missing environment variables: {missing_vars}")

        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        self.db_aoi_endpoint = os.getenv('ASTRA_DB_ENDPOINT')
        self.db_application_token = os.getenv('ASTRA_DB_APPLICATION_TOKEN')
        self.db_keyspace = os.getenv('ASTRA_DB_KEYSPACE')

    def load_retriever(self):
        pass

    def call_retriever(self, user_query: str):
        pass
