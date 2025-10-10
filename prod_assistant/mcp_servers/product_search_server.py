# prod_assistant/mcp_servers/product_search_server.py

"""
================================================================================
 PulseFlow – MCP Server: Product + Web Search Handler
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : mcp_servers.product_search_server
- Version     : 1.0.0
- Created on  : 2025-10-10
- Last Updated: 2025-10-10
- Environment : Python 3.11.13 | MCP | AstraDB | DuckDuckGo | StructLog
================================================================================

This module implements a **hybrid search MCP server** that powers the
retrieval layer for PulseFlow’s `AgenticRAG` pipeline.

Core Responsibilities
---------------------
- Serve product retrieval queries (AstraDB vector store).
- Handle fallback web searches (DuckDuckGo) when local recall is low.
- Maintain per-request observability via trace_id propagation.
- Write structured JSON logs per trace (for both retriever & web search).

Workflow Topology
-----------------
Client (AgenticRAG) → MCP Transport (FastMCP) → Product Search Server
→ Retriever (AstraDB) or DuckDuckGo → JSON Response

External Integrations
---------------------
- **AstraDB** — Vector store for product embeddings.
- **DuckDuckGo** — External fallback search API.
- **StructLog** — Structured trace logging.
- **FastMCP** — MCP protocol for async client–server communication.
- **LangChain Retriever** — Handles hybrid retrieval via OpenAI embeddings.

Changelog
---------
v1.0.0 (2025-10-10)
    • Initial stable MCP hybrid_search server.
    • Added trace_id propagation for observability.
    • Structured stage-based startup logs.
    • Exception-safe retriever and search tool execution.

License
-------
Copyright © 2025 Bhagwat Chate.
This code is part of the **PulseFlow** system under the personal projects umbrella.
All rights reserved.
"""

# ---------------------------------------------------------------------
# Bootstrap Initialization
# ---------------------------------------------------------------------
from prod_assistant.core.bootstrap import bootstrap_app
bootstrap_app()  # ensures LOGGER + CONFIG are initialized

import os
import json
import platform
from typing import Optional, Dict, Any
from mcp.server.fastmcp import FastMCP
from langchain_community.tools import DuckDuckGoSearchRun
from prod_assistant.retriever.retrieval import Retriever
from prod_assistant.core.globals import TRACE_ID, LOGGER
from prod_assistant.core.server_logger import get_mcp_logger, log_stage


# ---------------------------------------------------------------------
# MCP Server + Tool Initialization
# ---------------------------------------------------------------------
mcp = FastMCP("hybrid_search")
retriever_obj = Retriever()
retriever = retriever_obj.load_retriever()
duckduckgo = DuckDuckGoSearchRun()


# ---------------------------------------------------------------------
# Helper: Set trace context
# ---------------------------------------------------------------------
def _apply_trace_context(trace_id: Optional[str]):
    """
    Safely propagate the trace_id from client to the MCP context.

    Parameters
    ----------
    trace_id : Optional[str]
        Trace identifier received from the AgenticRAG client.

    Notes
    -----
    If no trace_id is provided, logs warning and continues using
    default context ('no-trace-id').
    """
    try:
        if trace_id:
            TRACE_ID.set(trace_id)
            if LOGGER:
                LOGGER.info("Trace propagated from client", trace_id=trace_id)
        else:
            if LOGGER:
                LOGGER.warning("No trace_id found in MCP payload — using default context")
    except Exception as e:
        if LOGGER:
            LOGGER.error("Trace context application failed", error=str(e))


# ---------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------
@mcp.tool()
async def get_product_info(payload: Dict[str, Any]) -> str:
    """
    Retrieve product information from AstraDB retriever.

    Parameters
    ----------
    payload : dict
        Expected keys:
            - query: str  → product or keyword to search
            - trace_id: str  → client trace identifier for logging correlation

    Returns
    -------
    str
        JSON string containing retrieved documents with metadata.
    """
    query = payload.get("query")
    trace_id = payload.get("trace_id")
    _apply_trace_context(trace_id)
    logger = get_mcp_logger(trace_id)

    try:
        logger.info("MCP[get_product_info]: Retrieval started", query=query, trace_id=trace_id)

        docs = retriever.invoke(query)
        logger.info("MCP[get_product_info]: Retrieved docs", count=len(docs), trace_id=trace_id)

        if not docs:
            return json.dumps([])

        result = [{"page_content": d.page_content, "metadata": d.metadata or {}} for d in docs]

        logger.info("MCP[get_product_info]: Completed successfully", trace_id=trace_id)
        return json.dumps(result)

    except Exception as e:
        logger.error("MCP[get_product_info]: Retrieval failed", error=str(e), trace_id=trace_id)
        return json.dumps([{"page_content": f"Error: {str(e)}", "metadata": {}}])


@mcp.tool()
async def web_search(payload: Dict[str, Any]) -> str:
    """
    Perform web search using DuckDuckGo for fallback results.

    Parameters
    ----------
    payload : dict
        Expected keys:
            - query: str  → search query text
            - trace_id: str  → client trace identifier

    Returns
    -------
    str
        Text or JSON result from web search API.
    """
    query = payload.get("query")
    trace_id = payload.get("trace_id")
    _apply_trace_context(trace_id)
    logger = get_mcp_logger(trace_id)

    try:
        logger.info("MCP[web_search]: Search started", query=query, trace_id=trace_id)
        result = duckduckgo.run(query)
        logger.info("MCP[web_search]: Search complete", trace_id=trace_id)
        return result or "No data from web"
    except Exception as e:
        logger.error("MCP[web_search]: Search failed", error=str(e), trace_id=trace_id)
        return f"Error during web search: {str(e)}"


# ---------------------------------------------------------------------
# Server Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    try:
        env = os.getenv("PULSEFLOW_APP_ENV", "dev")

        # Startup lifecycle logs
        log_stage(stage="startup", event="Starting MCP hybrid_search server", mode="Product + Web Search", env=env)
        log_stage(stage="retriever_init", event="Initializing Retriever", source="AstraDB + OpenAI Embeddings", env=env)
        log_stage(stage="db_connect", event="Connecting AstraDB", keyspace="pulseflow_keyspace", collection="pulseflow_collection", env=env)
        log_stage(stage="web_search_tool", event="DuckDuckGo tool initialized", env=env)

        # Start server
        mcp.run(transport="stdio")

        log_stage(stage="server_ready", event="MCP hybrid_search server started and ready to serve requests", transport="stdio", env=env)

    except Exception as e:
        log_stage(stage="fatal_error", event="MCP server startup failed", error=str(e), env=os.getenv("PULSEFLOW_APP_ENV", "dev"))
        raise
