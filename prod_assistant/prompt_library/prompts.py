# prod_assistant/prompt_library/prompts.py

from enum import Enum
from typing import Dict
import string


class PromptType(str, Enum):
    PRODUCT_BOT = "product_bot"
    ROUTER_BOT = "router_bot"
    # REVIEW_BOT = "review_bot"
    # COMPARISON_BOT = "comparison_bot"


class PromptTemplate:
    def __init__(self, template: str, description: str = "", version: str = "v1"):
        self.template = template.strip()
        self.description = description
        self.version = version

    def format(self, **kwargs) -> str:
        # Validate placeholders before formatting
        missing = [
            f for f in self.required_placeholders() if f not in kwargs
        ]
        if missing:
            raise ValueError(f"Missing placeholders: {missing}")
        return self.template.format(**kwargs)

    def required_placeholders(self):
        return [field_name for _, field_name, _, _ in string.Formatter().parse(self.template) if field_name]


# Central Registry
PROMPT_REGISTRY: Dict[PromptType, PromptTemplate] = {
    PromptType.PRODUCT_BOT: PromptTemplate(
        """
        You are an expert EcommerceBot specialized in product recommendations and handling customer queries.
        Analyze the provided product titles, ratings, and reviews to provide accurate, helpful responses.
        Stay relevant to the context, and keep your answers concise and informative.

        CONTEXT:
        {context}

        QUESTION: {question}

        YOUR ANSWER:
        """,
        description="Handles ecommerce QnA & product recommendation flows"
    ),
    PromptType.ROUTER_BOT: PromptTemplate(
        """
        You are a router agent in an Agentic RAG workflow.

        Priority rules:
        1. If the query is about a product (price, reviews, features, comparison, etc.), 
           always choose **retriever** first.
        2. If retriever has no relevant results, fallback to **web_search**.
        3. If the query is small-talk or unrelated to products, answer **direct**.

        Query: {query}

        Options:
        - "retriever"
        - "web_search"
        - "direct"

        Return ONLY one option.
        """,
        description="Decides whether to use retriever, web search, or answer directly"
    ),
}
