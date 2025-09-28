# file path: prod_assistant/mcp_servers/client.py

import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient


async def main():
    """
    MCP Client: Product Search Example

    This client connects to the `hybrid_search` MCP server, which exposes
    product search capabilities (local retriever + web search fallback).
    It demonstrates how to:
      - Connect to the server via stdio
      - Discover available tools dynamically
      - Call `get_product_info` for local results
      - Fall back to `web_search` if no local results are available

    Workflow:
        1. Start the `product_search_server.py` MCP server (via stdio).
        2. Fetch tools using `client.get_tools()`.
        3. Call `get_product_info(query)` to check vector DB retrieval.
        4. If empty → call `web_search(query)` for live DuckDuckGo results.

    Returns:
        None
    """

    # Connect to hybrid_search MCP server
    client = MultiServerMCPClient(
        {
            "hybrid_search": {
                "command": "python",
                "args": [r"prod_assistant/mcp_servers/product_search_server.py"],
                "transport": "stdio",
            }
        }
    )

    # Discover tools exposed by the server
    tools = await client.get_tools()
    print("Available tools: ", [t.name for t in tools])

    # Select tools
    retriever_tool = next(t for t in tools if t.name == "get_product_info")
    web_tool = next(t for t in tools if t.name == "web_search")

    # Example query
    query = "What is the price of iPhone 15?"

    # Call retriever tool
    retriever_result = await retriever_tool.ainvoke({"query": query})
    print("Retriever Results:\n", retriever_result)

    # Fallback to web search if no local results
    if not retriever_result.strip() or "No local results" in retriever_result:
        print("\nNo local results, falling back to web search...\n")
        web_result = await web_tool.ainvoke({"query": query})
        print("Web Search Results:\n", web_result)


if __name__ == "__main__":
    asyncio.run(main())
