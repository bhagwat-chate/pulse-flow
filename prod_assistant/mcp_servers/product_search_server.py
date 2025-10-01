# prod_assistant/mcp_servers/product_search_server.py

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


def format_docs(docs) -> str:
    """Format retriever docs into readable context."""
    if not docs:
        return ""
    formatted_chunks = []
    for d in docs:
        meta = d.metadata or {}
        formatted = (
            f"Title: {meta.get('product_title', 'N/A')}\n"
            f"Price: {meta.get('price', 'N/A')}\n"
            f"Rating: {meta.get('rating', 'N/A')}\n"
            f"Reviews:\n{d.page_content.strip()}"
        )
        formatted_chunks.append(formatted)
    return "\n\n---\n\n".join(formatted_chunks)


@mcp.tool()
async def get_product_info(query: str) -> str:
    """
    LLM Tool: Retrieve structured product information from the local vector retriever.

    This tool is designed to be called by an LLM agent (e.g., LangChain, LangGraph, MCP client)
    when the model detects the user is asking about a product (e.g., "iPhone 15 Plus reviews",
    "Samsung Galaxy price", "Suggest me washing machine under 50,000 INR").

    Behavior:
      - Runs a semantic vector search (MMR) over the AstraDB collection.
      - Returns formatted product information (title, price, rating, reviews).
      - If no results are found, returns "No local results found." so that the
        calling LLM can decide to fallback to `web_search`.

    Args:
        query (str): Natural language product search query from the user.

    Returns:
        str: Human-readable product details (multi-line text). This string is
             intended for direct injection into the LLM's context or response.

    Usage in an LLM pipeline:
        - Call this tool when the user asks for product prices, reviews, or comparisons.
        - If the response contains "No local results found.", call the `web_search` tool.
    """
    try:
        docs = retriever.invoke(query)
        context = format_docs(docs)
        if not context.strip():
            return "No local results found."
        return context
    except Exception as e:
        return f"Error retrieving product info: {str(e)}"


@mcp.tool()
async def web_search(query: str) -> str:
    """
    LLM Tool: Perform a live web search for product information.

    This tool is intended as a fallback when `get_product_info` returns
    "No local results found." or when the LLM determines the user query
    is outside the scope of the local vector database.

    Behavior:
      - Executes a DuckDuckGo search for the given query.
      - Returns raw text snippets of results (titles, descriptions, context).
      - Provides up-to-date product information beyond the static local corpus.

    Args:
        query (str): Natural language product search query from the user.

    Returns:
        str: Web search result snippets. This may be noisy, so the calling LLM
             is responsible for summarizing, filtering, or reasoning over it.

    Usage in an LLM pipeline:
        - Call when local retrieval fails or when freshness/coverage requires web lookup.
        - Use results to enrich or cross-check responses with the latest market data.
    """
    try:
        return duckduckgo.run(query)
    except Exception as e:
        return f"Error during web search: {str(e)}"


# ---------- Run Server ----------
if __name__ == "__main__":
    mcp.run(transport="stdio")
