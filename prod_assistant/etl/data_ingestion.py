# prod_assistant/etl/data_ingestion.py

import os
import pandas as pd
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore
from prod_assistant.utils.model_loader import ModelLoader
from prod_assistant.core.config.config_dev import get_config


class DataIngestion:
    def __init__(self):
        print("initializing data ingestion pipeline...")
        self.model_loader = ModelLoader()
        self._load_env_variables()
        self.csv_path = self._get_csv_path()
        self.product_data = self._load_csv()
        self.config = get_config()

    def _load_env_variables(self):
        load_dotenv()

        required_vars = ['OPENAI_API_KEY', 'GOOGLE_API_KEY', 'ASTRA_DB_API_ENDPOINT', 'ASTRA_DB_APPLICATION_TOKEN', 'ASTRA_DB_KEYSPACE']
        missing_vars = [var for var in required_vars if os.getenv(var) is None]

        if missing_vars:
            raise EnvironmentError(f"missing environment variables: {missing_vars}")

        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.db_api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv('ASTRA_DB_KEYSPACE')

    def _get_csv_path(self):
        current_path = os.getcwd()
        csv_path = os.path.join(current_path, 'data', 'product_reviews.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")

        return csv_path

    def _load_csv(self):
        df = pd.read_csv(self.csv_path)
        expected_columns = {'product_id', "product_title", 'rating', 'total_reviews', 'price', 'top_reviews'}

        if not expected_columns.issubset(df.columns):
            raise ValueError(f"CSV must contain all the columns: {expected_columns}")

        return df

    def transform_data(self):
        """
        Transform product DataFrame into LangChain Document objects.

        Returns:
            List[Document]: List of product review documents with metadata.
        """
        documents = []

        for row in self.product_data.to_dict(orient="records"):
            review_text = str(row.get("top_reviews", "")).strip()
            if not review_text:  # skip missing/empty reviews
                continue

            metadata = {
                "product_id": row["product_id"],
                "product_title": row["product_title"],
                "rating": row["rating"],
                "total_reviews": row["total_reviews"],
                "price": row["price"],
            }
            # split multiple reviews by "||"
            reviews = [r.strip() for r in review_text.split("||") if r.strip()]
            for review in reviews:
                documents.append(Document(page_content=review, metadata=metadata))

        print(f"Transformed {len(documents)} product reviews into documents. Final document for DB {len}")
        return documents

    def chunk_reviews(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,  # ~500 chars (tune depending on reviews)
            chunk_overlap=10,  # overlap to preserve context
            separators=["||"]
        )
        new_docs = []
        for doc in documents:
            chunks = splitter.split_text(doc.page_content)
            for chunk in chunks:
                new_docs.append(
                    Document(page_content=chunk, metadata=doc.metadata)
                )
        return new_docs

    def store_in_vector_db(self, documents: List[Document]):

        doc_chunks = self.chunk_reviews(documents)
        print(f"chunks: {len(doc_chunks)} created from docs: {len(documents)}")

        collection_name = self.config['astra_db']['collection_name']
        vector_store = AstraDBVectorStore(
            embedding=self.model_loader.load_embeddings(),
            collection_name=collection_name,
            api_endpoint=self.db_api_endpoint,
            token=self.db_application_token,
            namespace=self.db_keyspace,
        )

        inserted_ids = vector_store.add_documents(doc_chunks)

        print(f"successfully loaded the documents into AstraDB: {len(doc_chunks)}")

        return vector_store, inserted_ids

    def run_pipeline(self):
        documents = self.transform_data()
        vector_store, _ = self.store_in_vector_db(documents)

        query = "what's is the iPhone 15 plus price in Pune, India?"
        results = vector_store.similarity_search(query)

        print(f"sample search results for the query: {query}")

        for res in results:
            print(f"content: {res.page_content}\nmetadata: {res.metadata}")


if __name__ == '__main__':
    ingestion = DataIngestion()
    ingestion.run_pipeline()
