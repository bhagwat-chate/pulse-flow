# prod_assistant/workflow/agentic_workflow_with_mcp_websearch.py

"""
================================================================================
 AgenticRAG Workflow (MCP + RAGAs)
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow: Multi-Agent Product Intelligence System
- Module      : agentic_workflow_with_mcp_websearch
- Version     : 1.0.0
- Created on  : 2025-10-06
- Last Updated: 2025-10-06
- Environment : Python 3.11.13 | LangChain v0.3 | LangGraph | MCP Adapter | RAGAs
================================================================================

This module defines the `AgenticRAG` class — an advanced multi-agent Retrieval-
Augmented Generation (RAG) pipeline that integrates LangGraph, MCP (Multi-Channel
Processing), and RAGAs evaluation for end-to-end product intelligence and
observability.

Core Responsibilities
---------------------
- Dynamically route user queries between retrieval, web search, or direct response agents.
- Perform multi-agent reasoning through structured LangGraph workflows.
- Retrieve relevant product and web data using asynchronously loaded MCP tools.
- Generate natural-language responses enhanced with LLM reasoning.
- Evaluate RAG quality (context precision, response relevancy) via RAGAs metrics.
- Emit structured JSON logs with `trace_id` for full observability and debugging.

Workflow Topology
-----------------
    Router → Retriever → Grader → Rewriter → WebSearch → Generator

External Integrations
---------------------
- **LangGraph** — orchestrates workflow execution and state transitions.
- **LangChain** — powers LLM and prompt interactions.
- **MCP Adapter** — connects external retrieval and web-search services.
- **RAGAs** — asynchronously evaluates retrieval quality.
- **LangSmith** — captures trace-level observability and metrics.

Changelog
---------
v1.0.0  (2025-10-06)
    • Initial stable integration of MCP tools, LangGraph orchestration, and
      asynchronous RAGAs evaluation.
    • Introduced structured JSON logging with `trace_id` correlation.
    • Modularized utility dependencies (`mcp_tool_loader`, `ragas_helper`, etc.).
    • Prepared for deployment within PulseFlow’s production observability stack.

License
-------
Copyright © 2025 Bhagwat Chate.
This code is part of the **PulseFlow** system under the personal projects umbrella.
All rights reserved.
"""

from prod_assistant.core.bootstrap import bootstrap_app
bootstrap_app()

import json
import asyncio
from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient

from prod_assistant.prompt_library.prompts import PROMPT_REGISTRY, PromptType
from prod_assistant.retriever.retrieval import Retriever
from prod_assistant.utils.model_loader import ModelLoader
from prod_assistant.utils.mcp_tool_loader import schedule_mcp_tool_loading
from prod_assistant.utils.ragas_helper import launch_ragas_evaluation
from prod_assistant.core.trace import get_trace_id
from prod_assistant.core.globals import LOGGER


class AgenticRAG:
    """
    Implements the Agentic Retrieval-Augmented Generation (RAG) pipeline using LangGraph and MCP tools.

    This class encapsulates a multi-agent reasoning flow designed to:
    - Dynamically decide between vector retrieval, web search, or direct LLM response.
    - Execute retrieval and generation workflows via MCP-based tools.
    - Apply RAGAs metrics asynchronously for real-time evaluation of retrieval fidelity.

    Attributes:
        retriever_obj (Retriever): Interface to the local vector store retriever.
        model_loader (ModelLoader): Loads appropriate LLMs and embedding models.
        llm (BaseLanguageModel): Active LLM instance used throughout the workflow.
        checkpointer (MemorySaver): In-memory checkpoint handler for LangGraph sessions.
        mcp_client (MultiServerMCPClient): Interface for managing multiple MCP servers.
        mcp_tools (list): Dynamically loaded MCP tool registry (e.g., `get_product_info`, `web_search`).
        workflow (StateGraph): The assembled LangGraph workflow definition.
        app (GraphExecutable): Compiled executable workflow with memory checkpointing.
    """

    class AgentState(TypedDict):
        """Defines the LangGraph agent state containing conversational message history."""
        messages: Annotated[Sequence[BaseMessage], add_messages]

    # ------------------------------------------------------
    # Initialization
    # ------------------------------------------------------
    def __init__(self):
        """
        Initialize the AgenticRAG pipeline and all its dependencies.

        Responsibilities:
            - Initialize retriever, model loader, and checkpoint manager.
            - Instantiate and register MCP client configuration.
            - Schedule asynchronous MCP tool loading for external data access.
            - Build and compile the LangGraph workflow definition.

        Note:
            MCP tools (product & web search) are loaded asynchronously via
            `schedule_mcp_tool_loading()` and will be ready after cold start delay.
        """
        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()
        self.mcp_client = MultiServerMCPClient({
            "hybrid_search": {
                "command": "python",
                "args": ["-m", "prod_assistant.mcp_servers.product_search_server"],
                "transport": "stdio",
            }
        })

        schedule_mcp_tool_loading(self.mcp_client)
        self.mcp_tools = []  # Populated asynchronously after startup

        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------
    # Node 1: Router
    # ------------------------------------------------------
    def _ai_assistant(self, state: AgentState):
        """
        Route the user query to the appropriate agent (retriever, web search, or direct).

        Args:
            state (AgentState): Current workflow state containing the latest user message.

        Returns:
            dict: Updated message sequence indicating the next agent or final response.

        Logic:
            - Uses a router LLM prompt to decide among {"retriever", "web_search", "direct"}.
            - Invalid or uncertain responses default to direct LLM answering.
        """
        LOGGER.info("Router node triggered", trace_id=get_trace_id())
        RouteDecision = Literal["retriever", "web_search", "direct"]
        last_message = state["messages"][-1].content

        router_prompt = ChatPromptTemplate.from_template(PROMPT_REGISTRY[PromptType.ROUTER_BOT].template)
        chain = router_prompt | self.llm | StrOutputParser()
        decision_str = chain.invoke({"query": last_message}).strip().lower().replace('"', '').replace("'", "")

        if decision_str not in ("retriever", "web_search", "direct"):
            LOGGER.warning("Invalid router decision", trace_id=get_trace_id(), decision=decision_str)
            decision_str = "direct"

        LOGGER.info("Router decision made", trace_id=get_trace_id(), decision=decision_str)

        if decision_str == "retriever":
            return {"messages": [HumanMessage(content="TOOL: retriever")]}
        elif decision_str == "web_search":
            return {"messages": [HumanMessage(content="TOOL: web_search")]}
        else:
            answer_prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Answer the user directly.\n\nQuestion: {question}\nAnswer:"
            )
            chain = answer_prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": last_message})
            return {"messages": [HumanMessage(content=response)]}

    # ------------------------------------------------------
    # Node 2: Retriever
    # ------------------------------------------------------
    def _vector_retriever(self, state: AgentState):
        """
        Execute product retrieval via the MCP `get_product_info` tool.

        Args:
            state (AgentState): Current agent state containing the query.

        Returns:
            dict: Message payload containing formatted retrieval results or MCP error.

        Process:
            - Calls the MCP product retrieval tool with the query and trace ID.
            - Parses and normalizes JSON-encoded results from the MCP response.
            - Structures product metadata (title, price, rating, reviews) into context text.
        """
        trace_id = get_trace_id()
        LOGGER.info("Retriever node triggered", trace_id=trace_id)
        query = state["messages"][-1].content

        tool = next((t for t in self.mcp_tools if t.name == "get_product_info"), None)
        if not tool:
            LOGGER.error("Retriever MCP tool missing", trace_id=trace_id)
            return {"messages": [HumanMessage(content="[MCP ERROR] get_product_info tool not available")]}

        payload = {"query": query, "trace_id": trace_id}
        result = asyncio.run(tool.ainvoke(payload))

        docs = []
        if isinstance(result, str):
            try:
                docs = json.loads(result)
            except Exception as e:
                LOGGER.warning("Failed to parse MCP result", trace_id=trace_id, error=str(e))
        elif isinstance(result, list):
            docs = result

        if not docs:
            context = "No local results found."
        else:
            context = "\n\n---\n\n".join([
                f"Title: {doc.get('metadata', {}).get('product_title', 'N/A')}\n"
                f"Price: {doc.get('metadata', {}).get('price', 'N/A')}\n"
                f"Rating: {doc.get('metadata', {}).get('rating', 'N/A')}\n"
                f"Reviews:\n{doc.get('page_content', '')}"
                for doc in docs
            ])

        LOGGER.info("Retriever completed", trace_id=trace_id, doc_count=len(docs))
        return {"messages": [HumanMessage(content=context)]}

    # ------------------------------------------------------
    # Node 3: Web Search
    # ------------------------------------------------------
    def _web_search(self, state: AgentState):
        """
        Execute a real-time web search using the MCP `web_search` tool.

        Args:
            state (AgentState): Current conversation state.

        Returns:
            dict: Message payload with summarized web search context.
        """
        trace_id = get_trace_id()
        LOGGER.info("WebSearch node triggered", trace_id=trace_id)
        query = state["messages"][-1].content

        tool = next((t for t in self.mcp_tools if t.name == "web_search"), None)
        if not tool:
            LOGGER.error("WebSearch MCP tool missing", trace_id=trace_id)
            return {"messages": [HumanMessage(content="[MCP ERROR] web_search tool not available")]}

        payload = {"query": query, "trace_id": trace_id}
        result = asyncio.run(tool.ainvoke(payload))
        context = result if result else "No data from web"

        LOGGER.info("WebSearch completed", trace_id=trace_id)
        return {"messages": [HumanMessage(content=context)]}

    # ------------------------------------------------------
    # Node 4: Grader
    # ------------------------------------------------------
    def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter"]:
        """
        Evaluate document relevance to decide the next step (generation or rewrite).

        Args:
            state (AgentState): State containing question and retrieved context.

        Returns:
            Literal["generator", "rewriter"]: Workflow decision.
        """
        trace_id = get_trace_id()
        LOGGER.info("Grader node triggered", trace_id=trace_id)

        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = PromptTemplate(
            template="You are a grader. Question: {question}\nDocs: {docs}\nAre docs relevant? yes/no",
            input_variables=["question", "docs"],
        )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "docs": docs})

        route = "generator" if "yes" in score.lower() else "rewriter"
        LOGGER.info("Grader decision", trace_id=trace_id, route=route)
        return route

    # ------------------------------------------------------
    # Node 5: Generator
    # ------------------------------------------------------
    def _generate(self, state: AgentState):
        """
        Generate the final user-facing response using contextual information.

        Args:
            state (AgentState): Agent state containing question and context.

        Returns:
            dict: Message payload with generated response text.

        Behavior:
            - Combines context and query using a product-optimized generation prompt.
            - Triggers asynchronous RAGAs evaluation (`launch_ragas_evaluation`) for quality tracking.
        """
        trace_id = get_trace_id()
        LOGGER.info("Generator node triggered", trace_id=trace_id)
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = ChatPromptTemplate.from_template(PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template)
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": docs, "question": question})

        launch_ragas_evaluation(question, response, docs)
        LOGGER.info("Generator completed", trace_id=trace_id)
        return {"messages": [HumanMessage(content=response)]}

    # ------------------------------------------------------
    # Node 6: Rewriter
    # ------------------------------------------------------
    def _rewrite(self, state: AgentState):
        """
        Rewrite unclear user queries into a search-engine-optimized version.

        Args:
            state (AgentState): Workflow state with the original query.

        Returns:
            dict: Message payload containing rewritten query.
        """
        trace_id = get_trace_id()
        LOGGER.info("Rewriter node triggered", trace_id=trace_id)
        question = state["messages"][0].content

        prompt = ChatPromptTemplate.from_template(
            "Rewrite this query to be clearer for search engines.\n\nQuery: {question}\nRewritten Query:"
        )
        chain = prompt | self.llm | StrOutputParser()
        new_q = chain.invoke({"question": question})

        LOGGER.info("Rewriter completed", trace_id=trace_id)
        return {"messages": [HumanMessage(content=new_q.strip())]}

    # ------------------------------------------------------
    # Workflow Definition
    # ------------------------------------------------------
    def _build_workflow(self):
        """
        Assemble and return the LangGraph workflow definition.

        Returns:
            StateGraph: Fully connected graph defining the execution pipeline.
        """
        LOGGER.info("Building LangGraph workflow", trace_id=get_trace_id())
        workflow = StateGraph(self.AgentState)

        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)
        workflow.add_node("WebSearch", self._web_search)

        workflow.add_edge(START, "Assistant")
        workflow.add_conditional_edges(
            "Assistant",
            lambda s: "Retriever" if "TOOL" in s["messages"][-1].content else END,
            {"Retriever": "Retriever", END: END},
        )
        workflow.add_conditional_edges(
            "Retriever",
            self._grade_documents,
            {"generator": "Generator", "rewriter": "Rewriter"},
        )
        workflow.add_edge("Generator", END)
        workflow.add_edge("Rewriter", "WebSearch")
        workflow.add_edge("WebSearch", "Generator")

        LOGGER.info("Workflow build complete", trace_id=get_trace_id())
        return workflow

    # ------------------------------------------------------
    # Public Entry
    # ------------------------------------------------------
    # def run(self, query: str, thread_id: str = "default_thread", namespace: str = "agenticrag") -> str:
    #     """
    #     Execute the entire AgenticRAG workflow for a given user query.
    #
    #     Args:
    #         query (str): Natural language user query or product-related question.
    #         thread_id (str, optional): Thread/session identifier for checkpointing.
    #
    #     Returns:
    #         str: Final generated response text.
    #     """
    #
    #     trace_id = get_trace_id()
    #
    #     LOGGER.info("Workflow invoked", trace_id=trace_id, query=query, namespace=namespace)
    #
    #     result = self.app.invoke(
    #         {"messages": [HumanMessage(content=query)]},
    #         config={"configurable": {"thread_id": thread_id, "checkpoint_ns": namespace}},
    #     )
    #
    #     final_response = result["messages"][-1].content
    #     LOGGER.info("Workflow completed", trace_id=trace_id)
    #
    #     return final_response

    async def run(self, query: str, thread_id: str = "default_thread") -> str:
        """Run the workflow for a given query and return the final answer."""
        result = await self.app.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        return result["messages"][-1].content


if __name__ == '__main__':
    # query = "what is the iPhone 15 plus price in India"
    # thread_id = 'BHAGWAT'
    # namespace = 'agenticrag'
    # agent = AgenticRAG()
    # agent.run(query, thread_id, namespace)

    agent = AgenticRAG()

    # Session 1
    agent.run("What is iPhone 15 Plus price?", thread_id="session_01")

    # Session 2 – same thread_id, resumed conversation
    agent.run("Compare it with Samsung S25 Ultra", thread_id="session_01")

