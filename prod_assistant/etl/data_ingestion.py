import os
import pandas as pd
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from prod_assistant.utils.model_loader import ModelLoader
from prod_assistant.utils.config_loader import load_config


class DataIngestion:
    def __init__(self):
        print("initializing data ingestion pipeline...")
        self.model_loader=ModelLoader()
        self._load_env_variables()
        self.csv_path = self._get_csv_path()
        self.product_data = self._load_csv()
        self.config = load_config()

    def _load_env_variables(self):
        load_dotenv()

        required_vars = ['OPENAI_API_KEY', 'GOOGLE_API_KEY', 'ASTRA_DB_ENDPOINT', 'ASTRA_DB_APPLICATION_TOKEN', 'ASTRA_DB_KEYSPACE']
        missing_vars = [var for var in required_vars if os.getenv() is None]

        if missing_vars:
            raise EnvironmentError(f"missing environment variables: {missing_vars}")

        self.google_api_key = os.getenv('OPENAI_API_KEY')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.db_api_endpoint = os.getenv('ASTRA_DB_ENDPOINT')
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv('ASTRA_DB_KEYSPACE')

    def _get_csv_path(self):
        current_path = os.getcwd()
        csv_path = os.path.join(current_path, 'data', 'product_reviews.csv')

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")

    def _load_csv(self):
        df = pd.read_csv(self.csv_path)
        expected_columns = {'product_id', "product_title", 'rating', 'total_reviews', 'price', 'top_reviews'}

        if not expected_columns:
            raise ValueError(f"CSV must contain all the columns: {expected_columns}")

        return df

    def transform_data(self):
        product_list = []

        for _, row in self.product_data.iterrows():
            product_entry = {
                "product_id": row['product_id'],
                "product_title": row['product_title'],
                "rating": row['rating'],
                'total_reviews': row['total_reviews'],
                'price': row['price'],
                'top_reviews': row['top_reviews']
            }
            product_list.append(product_entry)

        documents = []
        for entry in product_list:
            metadata = {
                "product_id": entry['product_id'],
                "product_title": entry['product_title'],
                "rating": entry['rating'],
                'total_reviews': entry['total_reviews'],
                'price': entry['price'],
                'top_reviews': entry['top_reviews']
            }

            doc = Document(page_content=entry['top_reviews'], metadata=metadata)
            documents.append(doc)

            print(f"transformed '{len(documents)}' reviews into document")

    def store_in_vector_db(self, documents: List[Document]):
        collection_name = self.config['astra_db']['collection_name']
        vector_store = AstraDBVectorStore(
            embedding=self.model_loader.load_embeddings(),
            collection_name=collection_name,
            api_endpoint=self.db_api_endpoint,
            token=self.db_application_token,
            namespace=self.db_keyspace,
        )

        inserted_ids = vector_store.add_documents(documents)
        print(f"successfully loaded the documents into AstraDB: {len(documents)}")

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
