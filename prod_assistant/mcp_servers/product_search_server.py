# prod_assistant/mcp_servers/product_search_server.py

# --- Bootstrap first ---
from prod_assistant.core.bootstrap import bootstrap_app
bootstrap_app()  # ensures LOGGER and CONFIG initialized

import sys
import json
from mcp.server.fastmcp import FastMCP
from prod_assistant.retriever.retrieval import Retriever
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize MCP server
mcp = FastMCP("hybrid_search")

# Load retriever once
retriever_obj = Retriever()
retriever = retriever_obj.load_retriever()

# LangChain DuckDuckGo tool
duckduckgo = DuckDuckGoSearchRun()


# ---------- MCP Tools ----------


@mcp.tool()
async def get_product_info(query: str) -> str:
    """
    Retrieve product info and return as JSON string.
    Client must parse it back into list[dict].
    """
    try:
        docs = retriever.invoke(query)

        print(f"[DEBUG SERVER] Retrieved {len(docs)} docs for query: {query}", file=sys.stderr)

        if not docs:
            return json.dumps([])

        result = [
            {"page_content": d.page_content, "metadata": d.metadata or {}}
            for d in docs
        ]
        return json.dumps(result)

    except Exception as e:
        return json.dumps([{"page_content": f"Error: {str(e)}", "metadata": {}}])


@mcp.tool()
async def web_search(query: str) -> str:
    """
    Perform a web search using DuckDuckGo when local retriever has no relevant results.

    Args:
        query (str): User query for general web search.

    Returns:
        str: Search results as text, or an error message if search fails.
    """
    try:
        return duckduckgo.run(query)
    except Exception as e:
        return f"Error during web search: {str(e)}"


# ---------- Run Server ----------
if __name__ == "__main__":
    mcp.run(transport="stdio")
