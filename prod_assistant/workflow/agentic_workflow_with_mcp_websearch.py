# prod_assistant/workflow/agentic_workflow_with_mcp_websearch.py
"""
================================================================================
 AgenticRAG Workflow (MCP + RAGAs)
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : agentic_workflow_with_mcp_websearch
- Version     : 1.0.1
- Created on  : 2025-10-06
- Last Updated: 2025-10-10
- Environment : Python 3.11.13 | LangChain v0.3 | LangGraph | MCP Adapter | RAGAs
================================================================================

Overview
--------
Defines the `AgenticRAG` class — a structured multi-agent RAG pipeline that
combines LangGraph orchestration, MCP-based retrieval, and asynchronous RAGAs
evaluation for real-time product intelligence and observability.

Responsibilities
----------------
- Dynamically route user queries between vector retrieval, web search, or direct LLM reasoning.
- Execute hybrid retrieval pipelines using MCP tools.
- Manage conversation state and checkpoints via LangGraph.
- Evaluate context precision and response relevancy asynchronously using RAGAs.
- Provide structured JSON logging with trace correlation for every workflow node.

External Integrations
---------------------
- LangGraph: Graph-based agent orchestration.
- LangChain: LLM + prompt abstractions.
- MCP: External tool adapters for hybrid retrieval & web search.
- RAGAs: Post-response evaluation.
- LangSmith: Traceable observability integration.

License
-------
© 2025 Bhagwat Chate. All Rights Reserved.
"""

# ---------------------------------------------------------------------------
# Core Imports and Bootstrap
# ---------------------------------------------------------------------------
from prod_assistant.core.globals import LOGGER
from prod_assistant.core.bootstrap import bootstrap_app

# Safe bootstrap (idempotent)
if not LOGGER:
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

from prod_assistant.utils.mcp_tool_loader import schedule_mcp_tool_loading
from prod_assistant.prompt_library.prompts import PROMPT_REGISTRY, PromptType
from prod_assistant.retriever.retrieval import Retriever
from prod_assistant.utils.model_loader import ModelLoader
from prod_assistant.utils.ragas_helper import launch_ragas_evaluation
from prod_assistant.core.trace import get_trace_id
from prod_assistant.core.globals import LOGGER
from prod_assistant.exception.custom_exception import ProductAssistantException


class AgenticRAG:
    """
    Core Agentic Retrieval-Augmented Generation pipeline using LangGraph + MCP.

    Encapsulates end-to-end agent reasoning flow across multiple nodes:
    Router → Retriever → Grader → Rewriter → WebSearch → Generator.
    """

    class AgentState(TypedDict):
        """LangGraph agent state schema containing sequential messages."""
        messages: Annotated[Sequence[BaseMessage], add_messages]

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def __init__(self):
        """Initialize all components of AgenticRAG pipeline."""
        trace_id = get_trace_id()
        try:
            self.retriever_obj = Retriever()
            self.model_loader = ModelLoader()
            self.llm = self.model_loader.load_llm()
            self.checkpointer = MemorySaver()

            # Initialize MCP client configuration
            self.mcp_client = MultiServerMCPClient({
                "hybrid_search": {
                    "command": "python",
                    "args": ["-m", "prod_assistant.mcp_servers.product_search_server"],
                    "transport": "stdio",
                }
            })

            # # Preload MCP tools synchronously to avoid missing tool warnings
            # try:
            #     self.mcp_tools = asyncio.run(self.mcp_client.get_tools())
            #     tool_names = [t.name for t in self.mcp_tools]
            #     LOGGER.info("MCP tools loaded successfully", trace_id=trace_id, tools=tool_names)
            # except Exception as e:
            #     LOGGER.warning("MCP tool preloading failed", trace_id=trace_id, error=str(e))
            #     self.mcp_tools = []

            # Initialize placeholder
            self.mcp_tools = []

            # Schedule background async loader
            schedule_mcp_tool_loading(self.mcp_client)

            # Build LangGraph workflow
            self.workflow = self._build_workflow()
            self.app = self.workflow.compile(checkpointer=self.checkpointer)

            LOGGER.info("AgenticRAG initialized successfully", trace_id=trace_id)

        except Exception as e:
            LOGGER.error("Failed to initialize AgenticRAG", trace_id=trace_id, error=str(e))
            raise ProductAssistantException("AgenticRAG initialization failed", e)

    # ------------------------------------------------------------------
    # Node 1: Router
    # ------------------------------------------------------------------
    def _ai_assistant(self, state: AgentState):
        """Determine which agent should handle the user query."""
        trace_id = get_trace_id()
        try:
            LOGGER.info("Router node triggered", trace_id=trace_id)
            last_message = state["messages"][-1].content

            router_prompt = ChatPromptTemplate.from_template(PROMPT_REGISTRY[PromptType.ROUTER_BOT].template)
            chain = router_prompt | self.llm | StrOutputParser()
            decision = chain.invoke({"query": last_message}).strip().lower().replace('"', '').replace("'", "")

            if decision not in ("retriever", "web_search", "direct"):
                LOGGER.warning("Invalid router decision; defaulting to direct", trace_id=trace_id, decision=decision)
                decision = "direct"

            LOGGER.info("Router decision made", trace_id=trace_id, decision=decision)

            if decision == "retriever":
                return {"messages": [HumanMessage(content="TOOL: retriever")]}
            elif decision == "web_search":
                return {"messages": [HumanMessage(content="TOOL: web_search")]}
            else:
                answer_prompt = ChatPromptTemplate.from_template(
                    "You are a helpful assistant. Answer directly.\nQuestion: {question}\nAnswer:"
                )
                chain = answer_prompt | self.llm | StrOutputParser()
                response = chain.invoke({"question": last_message})
                return {"messages": [HumanMessage(content=response)]}

        except Exception as e:
            LOGGER.error("Router node failed", trace_id=trace_id, error=str(e))
            return {"messages": [HumanMessage(content="Routing error occurred. Defaulting to direct response.")]}

    # ------------------------------------------------------------------
    # Node 2: Retriever
    # ------------------------------------------------------------------
    def _vector_retriever(self, state: AgentState):
        """Fetch product data using MCP retrieval tools."""
        trace_id = get_trace_id()
        try:
            LOGGER.info("Retriever node triggered", trace_id=trace_id)
            query = state["messages"][-1].content

            tool = next((t for t in self.mcp_tools if t.name == "get_product_info"), None)
            if not tool:
                LOGGER.error("MCP tool get_product_info not available", trace_id=trace_id)
                return {"messages": [HumanMessage(content="[MCP ERROR] get_product_info not available")]}

            result = asyncio.run(tool.ainvoke({"query": query, "trace_id": trace_id}))

            # Parse JSON or structured list
            docs = []
            if isinstance(result, str):
                try:
                    docs = json.loads(result)
                except Exception:
                    LOGGER.warning("Failed to parse MCP result", trace_id=trace_id)
            elif isinstance(result, list):
                docs = result

            if not docs:
                context = "No local results found."
            else:
                context = "\n\n---\n\n".join([
                    f"Title: {d.get('metadata', {}).get('product_title', 'N/A')}\n"
                    f"Price: {d.get('metadata', {}).get('price', 'N/A')}\n"
                    f"Rating: {d.get('metadata', {}).get('rating', 'N/A')}\n"
                    f"Reviews:\n{d.get('page_content', '')}"
                    for d in docs
                ])

            LOGGER.info("Retriever node completed", trace_id=trace_id, doc_count=len(docs))
            return {"messages": [HumanMessage(content=context)]}

        except Exception as e:
            LOGGER.error("Retriever node failed", trace_id=trace_id, error=str(e))
            return {"messages": [HumanMessage(content="[Retriever Error] Failed to retrieve product info.")]}

    # ------------------------------------------------------------------
    # Node 3: Web Search
    # ------------------------------------------------------------------
    def _web_search(self, state: AgentState):
        """Execute real-time web search via MCP adapter."""
        trace_id = get_trace_id()
        try:
            LOGGER.info("WebSearch node triggered", trace_id=trace_id)
            query = state["messages"][-1].content

            tool = next((t for t in self.mcp_tools if t.name == "web_search"), None)
            if not tool:
                LOGGER.error("MCP tool web_search not available", trace_id=trace_id)
                return {"messages": [HumanMessage(content="[MCP ERROR] web_search not available")]}

            result = asyncio.run(tool.ainvoke({"query": query, "trace_id": trace_id}))
            context = result if result else "No web results available."

            LOGGER.info("WebSearch node completed", trace_id=trace_id)
            return {"messages": [HumanMessage(content=context)]}

        except Exception as e:
            LOGGER.error("WebSearch node failed", trace_id=trace_id, error=str(e))
            return {"messages": [HumanMessage(content="[WebSearch Error] Failed to retrieve search results.")]}

    # ------------------------------------------------------------------
    # Node 4: Grader
    # ------------------------------------------------------------------
    def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter"]:
        """Assess document relevancy and decide workflow path."""
        trace_id = get_trace_id()
        try:
            LOGGER.info("Grader node triggered", trace_id=trace_id)
            question = state["messages"][0].content
            docs = state["messages"][-1].content

            prompt = PromptTemplate(
                template="You are a grader.\nQuestion: {question}\nDocs: {docs}\nAre docs relevant? yes/no",
                input_variables=["question", "docs"],
            )
            chain = prompt | self.llm | StrOutputParser()
            score = chain.invoke({"question": question, "docs": docs})

            route = "generator" if "yes" in score.lower() else "rewriter"
            LOGGER.info("Grader decision", trace_id=trace_id, route=route)
            return route

        except Exception as e:
            LOGGER.error("Grader node failed", trace_id=trace_id, error=str(e))
            return "generator"

    # ------------------------------------------------------------------
    # Node 5: Generator
    # ------------------------------------------------------------------
    def _generate(self, state: AgentState):
        """Generate the final product intelligence response."""
        trace_id = get_trace_id()
        try:
            LOGGER.info("Generator node triggered", trace_id=trace_id)
            question = state["messages"][0].content
            docs = state["messages"][-1].content

            prompt = ChatPromptTemplate.from_template(PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template)
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"context": docs, "question": question})

            launch_ragas_evaluation(question, response, docs)
            LOGGER.info("Generator node completed", trace_id=trace_id)
            return {"messages": [HumanMessage(content=response)]}

        except Exception as e:
            LOGGER.error("Generator node failed", trace_id=trace_id, error=str(e))
            return {"messages": [HumanMessage(content="[Generator Error] Unable to produce response.")]}

    # ------------------------------------------------------------------
    # Node 6: Rewriter
    # ------------------------------------------------------------------
    def _rewrite(self, state: AgentState):
        """Rewrite unclear or irrelevant user queries for better search."""
        trace_id = get_trace_id()
        try:
            LOGGER.info("Rewriter node triggered", trace_id=trace_id)
            question = state["messages"][0].content

            prompt = ChatPromptTemplate.from_template(
                "Rewrite this query for better search results.\nQuery: {question}\nRewritten Query:"
            )
            chain = prompt | self.llm | StrOutputParser()
            new_q = chain.invoke({"question": question}).strip()

            LOGGER.info("Rewriter node completed", trace_id=trace_id)
            return {"messages": [HumanMessage(content=new_q)]}

        except Exception as e:
            LOGGER.error("Rewriter node failed", trace_id=trace_id, error=str(e))
            return {"messages": [HumanMessage(content="[Rewriter Error] Failed to rewrite query.")]}

    # ------------------------------------------------------------------
    # Workflow Builder
    # ------------------------------------------------------------------
    def _build_workflow(self):
        """Construct the full LangGraph workflow definition."""
        trace_id = get_trace_id()
        try:
            LOGGER.info("Building LangGraph workflow", trace_id=trace_id)
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

            LOGGER.info("Workflow graph built successfully", trace_id=trace_id)
            return workflow

        except Exception as e:
            LOGGER.error("Workflow build failed", trace_id=trace_id, error=str(e))
            raise ProductAssistantException("Failed to build LangGraph workflow", e)

    # ------------------------------------------------------------------
    # Execution Entry Point
    # ------------------------------------------------------------------
    async def run(self, query: str, thread_id: str = "default_thread") -> str:
        """Execute the entire workflow asynchronously and return final answer."""
        trace_id = get_trace_id()
        LOGGER.info("AgenticRAG workflow started", trace_id=trace_id, query=query)
        try:
            result = await self.app.ainvoke(
                {"messages": [HumanMessage(content=query)]},
                config={"configurable": {"thread_id": thread_id}},
            )
            response = result["messages"][-1].content
            LOGGER.info("AgenticRAG workflow completed", trace_id=trace_id)
            return response

        except Exception as e:
            LOGGER.error("AgenticRAG workflow execution failed", trace_id=trace_id, error=str(e))
            return (
                "Sorry, an internal issue occurred while processing your query. "
                "Please retry or contact support with your trace ID."
            )


# ---------------------------------------------------------------------------
# Standalone Test Mode
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Development-only entry for manual workflow testing.
    """
    try:
        LOGGER.info("Starting AgenticRAG standalone workflow", trace_id="dev-mode")
        agent = AgenticRAG()
        asyncio.run(agent.run("What is the iPhone 15 Plus price?", thread_id="session_01"))
    except Exception as e:
        LOGGER.error("AgenticRAG standalone execution failed", error=str(e))
