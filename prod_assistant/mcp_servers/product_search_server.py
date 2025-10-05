# prod_assistant/mcp_servers/product_search_server.py

"""
MCP Server - Product Search
===========================

Purpose:
--------
Handles local product info retrieval (AstraDB) and web search (DuckDuckGo)
for the AgenticRAG client via MCP transport.

Enhancements:
-------------
✅ Accepts payload as dict: {"query": str, "trace_id": str}
✅ Propagates trace_id for unified observability
✅ Structured logging for both tools
"""

# --- Bootstrap first ---
from prod_assistant.core.bootstrap import bootstrap_app
bootstrap_app()  # ensures LOGGER + CONFIG are initialized

import sys
import json
from typing import Optional, Dict, Any
from mcp.server.fastmcp import FastMCP
from prod_assistant.retriever.retrieval import Retriever
from langchain_community.tools import DuckDuckGoSearchRun
from prod_assistant.core.globals import TRACE_ID, LOGGER
from prod_assistant.core.server_logger import get_mcp_logger

# ----------------------------------------------------------------------
# Initialize MCP Server + Core Components
# ----------------------------------------------------------------------
mcp = FastMCP("hybrid_search")

retriever_obj = Retriever()
retriever = retriever_obj.load_retriever()
duckduckgo = DuckDuckGoSearchRun()


# ----------------------------------------------------------------------
# Helper: Set trace context from incoming payload
# ----------------------------------------------------------------------
def _apply_trace_context(trace_id: Optional[str]):
    """Safely set TRACE_ID context for propagated client request."""
    if trace_id:
        TRACE_ID.set(trace_id)
        LOGGER.info("Trace propagated from client", trace_id=trace_id)
    else:
        LOGGER.warning("No trace_id found in MCP payload — using default context")


# ----------------------------------------------------------------------
# MCP Tools
# ----------------------------------------------------------------------
@mcp.tool()
async def get_product_info(payload: Dict[str, Any]) -> str:
    """
    Retrieve product info from AstraDB retriever.
    Expects payload: {"query": str, "trace_id": str}
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
    Perform a web search using DuckDuckGo when local retriever has no relevant results.
    Expects payload: {"query": str, "trace_id": str}
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


# ----------------------------------------------------------------------
# Run MCP Server
# ----------------------------------------------------------------------
if __name__ == "__main__":
    LOGGER.info("🚀 Starting MCP hybrid_search server (Product + Web Search)")
    mcp.run(transport="stdio")
