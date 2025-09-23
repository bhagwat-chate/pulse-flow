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
        Initialize and return a retriever from AstraDB Vector Store.

        Returns:
            VectorStoreRetriever: A retriever instance for semantic search.
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
            retriever_config = self.config.get("retriever", {})
            top_k = retriever_config.get("top_k", 3)
            self.retriever = self.vstore.as_retriever(search_kwargs={"k": top_k})

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
    retriever_obj = Retriever()
    user_query = "iPhone 15"
    output = retriever_obj.call_retriever(user_query)
    print(output)
