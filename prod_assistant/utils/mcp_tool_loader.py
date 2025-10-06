# prod_assistant/utils/mcp_tool_loader.py
import asyncio
from prod_assistant.core.globals import LOGGER
from prod_assistant.core.trace import get_trace_id


async def load_mcp_tools_async(mcp_client):
    """Asynchronously loads MCP tools with robust error handling."""
    trace_id = get_trace_id()
    try:
        tools = await mcp_client.get_tools()
        tool_names = [t.name for t in tools]
        LOGGER.info("MCP tools loaded successfully", tools=tool_names, trace_id=trace_id)
        return tools
    except Exception as e:
        LOGGER.error("Failed to load MCP tools asynchronously", trace_id=trace_id, error=str(e))
        return []


def schedule_mcp_tool_loading(mcp_client):
    """
    Handles event loop safety.
    Runs async loading if possible; schedules it if already in a loop.
    """
    trace_id = get_trace_id()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(load_mcp_tools_async(mcp_client))
        else:
            loop.run_until_complete(load_mcp_tools_async(mcp_client))
    except Exception as e:
        LOGGER.error("Failed to start MCP tool loader", trace_id=trace_id, error=str(e))
