"""
================================================================================
 PulseFlow ETL – Product Data Ingestion Pipeline
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : etl.data_ingestion
- Version     : 1.0.0
- Created on  : 2025-10-10
- Last Updated: 2025-10-10
- Environment : Python 3.11.13 | LangChain | AstraDB | StructLog
================================================================================

This module implements the **DataIngestion** class responsible for transforming
raw product review datasets into vectorized LangChain documents and storing them
in AstraDB. It acts as the ETL (Extract–Transform–Load) component for PulseFlow’s
semantic product intelligence engine.

Core Responsibilities
---------------------
- Validate and load environment variables required for embedding and AstraDB.
- Load product review CSV data from local or mounted storage.
- Transform reviews into LangChain `Document` objects with structured metadata.
- Chunk large reviews for optimal vector storage and retrieval.
- Store transformed data into AstraDB Vector Store using `ModelLoader` embeddings.
- Validate ingestion by running a sample semantic search query.

Workflow Topology
-----------------
    CSV → DataFrame → Documents → Chunked Documents → AstraDB Vector Store

External Integrations
---------------------
- **ModelLoader** — Loads OpenAI embedding model from configuration.
- **AstraDBVectorStore** — Performs vector storage and retrieval operations.
- **LangChain Documents** — Encapsulates text and metadata for embedding.
- **dotenv / AWS Secrets** — Provides secure credentials for connections.

Changelog
---------
v1.0.0 (2025-10-10)
    • Initial version with environment validation and data ingestion pipeline.
    • Added chunking and semantic storage in AstraDB.
    • Introduced sample retrieval test for post-ingestion validation.

License
-------
Copyright © 2025 Bhagwat Chate.
This code is part of the **PulseFlow** system under the personal projects umbrella.
All rights reserved.
"""

from prod_assistant.core.bootstrap import bootstrap_app
bootstrap_app()

import os
import pandas as pd
from dotenv import load_dotenv
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore
from prod_assistant.utils.model_loader import ModelLoader
from prod_assistant.core.globals import get_config, LOGGER


class DataIngestion:
    """
    Handles ETL workflow for product reviews ingestion and storage
    into the AstraDB vector store for semantic retrieval.

    Steps:
        1. Load environment and configuration.
        2. Validate product review CSV.
        3. Transform into LangChain documents.
        4. Chunk and embed data for storage.
    """

    def __init__(self):
        try:
            LOGGER.info("Initializing DataIngestion pipeline")
            self.model_loader = ModelLoader()
            self._load_env_variables()
            self.csv_path = self._get_csv_path()
            self.product_data = self._load_csv()
            self.config = get_config()
            LOGGER.info("DataIngestion initialized successfully")
        except Exception as e:
            LOGGER.error("DataIngestion initialization failed", error=str(e))
            raise

    # ------------------------------------------------------------------
    # Environment Validation
    # ------------------------------------------------------------------
    def _load_env_variables(self):
        """
        Load and validate environment variables required for ingestion.

        Raises
        ------
        EnvironmentError
            If any required environment variable is missing.
        """
        load_dotenv()
        required_vars = [
            'OPENAI_API_KEY',
            'GOOGLE_API_KEY',
            'ASTRA_DB_API_ENDPOINT',
            'ASTRA_DB_APPLICATION_TOKEN',
            'ASTRA_DB_KEYSPACE'
        ]
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise EnvironmentError(f"Missing environment variables: {missing_vars}")

        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.db_api_endpoint = os.getenv('ASTRA_DB_API_ENDPOINT')
        self.db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
        self.db_keyspace = os.getenv('ASTRA_DB_KEYSPACE')

    # ------------------------------------------------------------------
    # Data Loading
    # ------------------------------------------------------------------
    def _get_csv_path(self) -> str:
        """
        Resolve and validate the CSV file path for product data.

        Returns
        -------
        str
            Path to the CSV file.

        Raises
        ------
        FileNotFoundError
            If the expected CSV file does not exist.
        """
        csv_path = os.path.join(os.getcwd(), 'data', 'product_reviews.csv')
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found at: {csv_path}")
        return csv_path

    def _load_csv(self) -> pd.DataFrame:
        """
        Load and validate the product review CSV file.

        Returns
        -------
        pd.DataFrame
            Loaded product review data.

        Raises
        ------
        ValueError
            If expected columns are missing from the CSV.
        """
        df = pd.read_csv(self.csv_path)
        expected_columns = {'product_id', "product_title", 'rating', 'total_reviews', 'price', 'top_reviews'}
        if not expected_columns.issubset(df.columns):
            raise ValueError(f"CSV must contain all the columns: {expected_columns}")
        return df

    # ------------------------------------------------------------------
    # Data Transformation
    # ------------------------------------------------------------------
    def transform_data(self) -> List[Document]:
        """
        Convert product reviews into LangChain Document objects.

        Returns
        -------
        List[Document]
            List of structured document objects for embedding.
        """
        try:
            documents = []
            for row in self.product_data.to_dict(orient="records"):
                review_text = str(row.get("top_reviews", "")).strip()
                if not review_text:
                    continue

                metadata = {
                    "product_id": row["product_id"],
                    "product_title": row["product_title"],
                    "rating": row["rating"],
                    "total_reviews": row["total_reviews"],
                    "price": row["price"],
                }

                reviews = [r.strip() for r in review_text.split("||") if r.strip()]
                for review in reviews:
                    documents.append(Document(page_content=review, metadata=metadata))

            LOGGER.info("Transformed product reviews into documents", count=len(documents))
            return documents
        except Exception as e:
            LOGGER.error("Data transformation failed", error=str(e))
            raise

    def chunk_reviews(self, documents: List[Document]) -> List[Document]:
        """
        Split long product reviews into smaller overlapping text chunks.

        Parameters
        ----------
        documents : List[Document]
            List of LangChain document objects.

        Returns
        -------
        List[Document]
            Chunked documents ready for embedding and vector storage.
        """
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=100,
                chunk_overlap=10,
                separators=["||"]
            )
            new_docs = []
            for doc in documents:
                for chunk in splitter.split_text(doc.page_content):
                    new_docs.append(Document(page_content=chunk, metadata=doc.metadata))
            LOGGER.info("Review chunking complete", total_chunks=len(new_docs))
            return new_docs
        except Exception as e:
            LOGGER.error("Review chunking failed", error=str(e))
            raise

    # ------------------------------------------------------------------
    # Data Storage
    # ------------------------------------------------------------------
    def store_in_vector_db(self, documents: List[Document]):
        """
        Store transformed product review documents into AstraDB vector store.

        Parameters
        ----------
        documents : List[Document]
            List of chunked review documents.

        Returns
        -------
        tuple
            (AstraDBVectorStore instance, list of inserted document IDs)
        """
        try:
            doc_chunks = self.chunk_reviews(documents)
            LOGGER.info("Preparing to load documents into AstraDB", chunk_count=len(doc_chunks))

            collection_name = self.config['astra_db']['collection_name']
            vector_store = AstraDBVectorStore(
                embedding=self.model_loader.load_embeddings(),
                collection_name=collection_name,
                api_endpoint=self.db_api_endpoint,
                token=self.db_application_token,
                namespace=self.db_keyspace,
            )

            inserted_ids = vector_store.add_documents(doc_chunks)
            LOGGER.info("Documents successfully loaded into AstraDB", total=len(doc_chunks))
            return vector_store, inserted_ids
        except Exception as e:
            LOGGER.error("Failed to store documents in AstraDB", error=str(e))
            raise

    # ------------------------------------------------------------------
    # Pipeline Runner
    # ------------------------------------------------------------------
    def run_pipeline(self):
        """
        Execute the complete ingestion pipeline end-to-end:
        1. Transform data
        2. Store in AstraDB
        3. Run sample similarity search
        """
        try:
            documents = self.transform_data()
            vector_store, _ = self.store_in_vector_db(documents)

            query = "what's the iPhone 15 Plus price in Pune, India?"
            results = vector_store.similarity_search(query)
            LOGGER.info("Sample search query executed", query=query, results=len(results))
        except Exception as e:
            LOGGER.error("Data ingestion pipeline execution failed", error=str(e))
            raise


if __name__ == '__main__':
    ingestion = DataIngestion()
    ingestion.run_pipeline()
