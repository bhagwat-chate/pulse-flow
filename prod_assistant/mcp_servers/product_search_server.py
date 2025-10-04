# prod_assistant/mcp_servers/product_search_server.py
#
# from mcp.server.fastmcp import FastMCP
# from prod_assistant.retriever.retrieval import Retriever
# from langchain_community.tools import DuckDuckGoSearchRun
#
# # Initialize MCP server
# mcp = FastMCP("hybrid_search")
#
# # Load retriever once
# retriever_obj = Retriever()
# retriever = retriever_obj.load_retriever()
#
# # LangChain DuckDuckGo tool
# duckduckgo = DuckDuckGoSearchRun()
#
# # ---------- Helpers ----------
# def format_docs(docs) -> str:
#     """Format retriever docs into readable context."""
#     if not docs:
#         return ""
#     formatted_chunks = []
#     for d in docs:
#         meta = d.metadata or {}
#         formatted = (
#             f"Title: {meta.get('product_title', 'N/A')}\n"
#             f"Price: {meta.get('price', 'N/A')}\n"
#             f"Rating: {meta.get('rating', 'N/A')}\n"
#             f"Reviews:\n{d.page_content.strip()}"
#         )
#         formatted_chunks.append(formatted)
#     return "\n\n---\n\n".join(formatted_chunks)
#
#
# # ---------- MCP Tools ----------
# # @mcp.tool()
# # async def get_product_info(query: str) -> str:
# #     """Retrieve product information for a given query from local retriever."""
# #     try:
# #         docs = retriever.invoke(query)
# #         context = format_docs(docs)
# #         if not context.strip():
# #             return "No local results found."
# #         return context
# #     except Exception as e:
# #         return f"Error retrieving product info: {str(e)}"
#
#
# @mcp.tool()
# async def get_product_info(query: str) -> list[dict]:
#     """
#     Retrieve product information for a given query from local retriever.
#     Returns raw documents as dicts, not just a string.
#     """
#     try:
#         docs = retriever.invoke(query)
#         if not docs:
#             return []
#         # Return JSON-safe list of dicts
#         return [
#             {
#                 "page_content": d.page_content,
#                 "metadata": d.metadata
#             }
#             for d in docs
#         ]
#     except Exception as e:
#         return [{"page_content": f"Error: {str(e)}", "metadata": {}}]
#
#
#
# @mcp.tool()
# async def web_search(query: str) -> str:
#     """
#     Perform a web search for the given query using DuckDuckGo.
#     Args:
#         query (str): The user query string.
#     Returns:
#         str: Search results in plain text format, or an error message.
#     """
#
#     try:
#         return duckduckgo.run(query)
#     except Exception as e:
#         return f"Error during web search: {str(e)}"
#
# # ---------- Run Server ----------
# if __name__ == "__main__":
#     mcp.run(transport="stdio")


# prod_assistant/mcp_servers/product_search_server.py
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
