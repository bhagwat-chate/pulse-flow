# prod_assistant/prompt_library/prompts.py

"""
================================================================================
 PulseFlow Prompt Library
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : prompt_library.prompts
- Version     : 1.0.0
- Created on  : 2025-10-10
- Last Updated: 2025-10-10
- Environment : Python 3.11.13 | LangChain | LangGraph
================================================================================

This module defines the **Prompt Library** — a centralized registry of reusable,
structured prompt templates used across PulseFlow’s AgenticRAG workflows.

Core Responsibilities
---------------------
- Maintain versioned prompt templates for different agent roles.
- Provide a safe, typed interface to validate and format prompts.
- Ensure placeholder consistency and enforce runtime validation.

Workflow Topology
-----------------
    AgenticRAG → PromptRegistry → PromptTemplate → LLM invocation

External Integrations
---------------------
- **AgenticRAG** – Consumes `PROMPT_REGISTRY` for routing and generation.
- **LangChain** – Uses formatted strings as LLM prompt inputs.
- **Router & Product Agents** – Retrieve prompts by `PromptType`.

Changelog
---------
v1.0.0 (2025-10-10)
    • Introduced `PromptType` enum and `PromptTemplate` class.
    • Added placeholder validation for robust formatting.
    • Registered product_bot and router_bot prompt templates.

License
-------
Copyright © 2025 Bhagwat Chate.
This code is part of the **PulseFlow** system under the personal projects umbrella.
All rights reserved.
"""

from enum import Enum
from typing import Dict, List
import string


# ---------------------------------------------------------------------
# Enum: Prompt Types
# ---------------------------------------------------------------------
class PromptType(str, Enum):
    """
    Enumeration of supported prompt categories used in PulseFlow agents.

    Attributes
    ----------
    PRODUCT_BOT : str
        Prompt for product recommendation and ecommerce QnA tasks.
    ROUTER_BOT : str
        Prompt for router agent to decide workflow routing.
    """
    PRODUCT_BOT = "product_bot"
    ROUTER_BOT = "router_bot"
    # REVIEW_BOT = "review_bot"
    # COMPARISON_BOT = "comparison_bot"


# ---------------------------------------------------------------------
# Class: Prompt Template
# ---------------------------------------------------------------------
class PromptTemplate:
    """
    Represents a versioned and validated prompt template.

    Parameters
    ----------
    template : str
        The raw prompt string containing placeholders (e.g., `{context}`, `{query}`).
    description : str, optional
        Description of the prompt’s purpose or agent role.
    version : str, optional
        Version tag for version control (default is "v1").

    Methods
    -------
    format(**kwargs)
        Safely injects runtime variables into the prompt.
    required_placeholders()
        Returns all required placeholders present in the template.
    """

    def __init__(self, template: str, description: str = "", version: str = "v1"):
        self.template = template.strip()
        self.description = description
        self.version = version

    # -----------------------------------------------------------------
    def format(self, **kwargs) -> str:
        """
        Format the template by first verifying that all required placeholders
        are present in `kwargs`, then calling Python's `str.format(**kwargs)`.

        Behavior
        --------
        - Raises `ValueError` listing any missing placeholder names returned by
          `required_placeholders()`.
        - Extra keys in `kwargs` are allowed; unused keys are ignored by `str.format`.
        - Performs no sanitization or type coercion of values.
        - Literal braces must be escaped in the template using `{{` and `}}`.

        Returns
        -------
        str
            The fully formatted prompt string ready for LLM input.
        """
        missing = [f for f in self.required_placeholders() if f not in kwargs]
        if missing:
            raise ValueError(f"Missing placeholders: {missing}")
        return self.template.format(**kwargs)

    def required_placeholders(self):
        """
        Parse the template and return the list of placeholder field names that
        require values.

        Behavior
        --------
        - Uses `string.Formatter().parse()` and collects the `field_name` elements.
        - Returns names in order of appearance; duplicates are preserved if a
          placeholder appears multiple times.
        - Ignores literal braces and any parsed segments where `field_name` is `None`.
        - Supports advanced format syntax (e.g., `{name!r:.2f}`); only the field
          name (`"name"`) is returned, not its conversion or format spec.

        Returns
        -------
        list[str]
            Names of placeholders found in the template (order preserved, may include duplicates).
        """
        return [
            field_name
            for _, field_name, _, _ in string.Formatter().parse(self.template)
            if field_name
        ]


# ---------------------------------------------------------------------
# Global Registry of Prompt Templates
# ---------------------------------------------------------------------
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
        description="Handles ecommerce QnA & product recommendation flows",
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
        description="Decides whether to use retriever, web search, or answer directly",
    ),
}
