# file path: prod_assistant/mcp_servers/product_search_server.py

from mcp.server.fastmcp import FastMCP
from prod_assistant.retriever.retrieval import Retriever
from langchain_community.tools import DuckDuckGoSearchRun

# Initialize MCP server
mcp = FastMCP("hybrid_search")

# Local retriever (vector DB / hybrid search)
retriever_obj = Retriever()
retriever = retriever_obj.load_retriever()

# Web search fallback
duckduckgo = DuckDuckGoSearchRun()


def format_docs(docs) -> str:
    """
    Format retriever documents into a human-readable context block.

    Args:
        docs (list): A list of Document objects returned by retriever.

    Returns:
        str: Concatenated and formatted product information, including
             title, price, rating, and snippet of reviews.
    """
    if not docs:
        return ""

    formatted_chunks = []

    for d in docs:
        meta = d.metadata or {}
        formatted = (
            f"Title: {meta.get('product_title', 'N/A')}\n"
            f"Price: {meta.get('price', 'N/A')}\n"
            f"Rating: {meta.get('rating', 'N/A')}\n"
            f"Reviews: {d.page_content.strip()}"
        )
        formatted_chunks.append(formatted)

    return "\n\n---\n\n".join(formatted_chunks)


@mcp.tool()
async def get_product_info(query: str) -> str:
    """
    MCP Tool: Retrieve product information from local retriever.

    This tool queries the local hybrid retriever (vector database / indexed
    product corpus) for relevant product details. If documents are found,
    they are formatted into a structured block showing title, price, rating,
    and reviews.

    Args:
        query (str): User's product search query.

    Returns:
        str: Formatted product details if available, or a notice indicating
             no local results and suggesting fallback to web search.

    Example:
        # await get_product_info("iPhone 15 price")
        # "Title: iPhone 15 Pro\nPrice: $999\nRating: 4.7\nReviews: Excellent camera..."
    """
    try:
        docs = retriever.invoke(query)
        context = format_docs(docs)

        if not context.strip():
            return "No local results found."

        return context

    except Exception as e:
        return f"Error retrieving product info: {e}"


@mcp.tool()
async def web_search(query: str) -> str:
    """
    MCP Tool: Perform a live web search for product information.

    This tool uses DuckDuckGo search to retrieve up-to-date product
    information when local retriever results are insufficient.

    Args:
        query (str): User's product search query.

    Returns:
        str: Raw text output from DuckDuckGo search results.

    Example:
        # await web_search("latest MacBook Air M3 price")
        # "Apple MacBook Air M3 – starting at $1199..."
    """
    try:
        return duckduckgo.run(query)
    except Exception as e:
        return f"Error during web search: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
