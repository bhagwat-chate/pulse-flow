# prod_assistant/utils/mcp_tool_loader.py
"""
================================================================================
 MCP Tool Loader Utilities
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : utils.mcp_tool_loader
- Version     : 1.0.5
- Created on  : 2025-10-10
- Environment : Python 3.11.13 | LangGraph | FastMCP
================================================================================

Description
-----------
Asynchronous loader for initializing predefined MCP tools within the
AgenticRAG orchestrator. This implementation does not depend on any
introspection APIs like `get_tools()` or `list_tools()`.

Architecture Context
--------------------
Layer:        Integration / MCP Adapter
Upstream:     AgenticRAG Orchestrator
Downstream:   FastMCP Servers (e.g., product_search_server)

Key Responsibilities
--------------------
• Register known MCP tools used in the orchestration layer.
• Handle async-safe initialization for concurrent workflows.
• Provide structured trace-aware observability.
• Prevent blocking of FastAPI event loop.

Engineering Standards
---------------------
• Non-blocking, async-safe design.
• Structured single-line logs only.
• Defensive exception handling — never crash.
• Compatible with FastMCP tool registration model.
"""

import asyncio
from prod_assistant.core.globals import LOGGER
from prod_assistant.core.trace import get_trace_id


# ======================================================================
# Async MCP Tool Loader
# ======================================================================
async def load_mcp_tools_async():
    """
    Asynchronously initialize and return a predefined list of MCP tool names.

    Behavior:
        - Returns a static list of registered tools known to this system.
        - Designed for systems using FastMCP without dynamic introspection.
        - Useful for consistency in multi-agent workflows.

    Returns:
        list[str]: List of MCP tool names.
    """
    trace_id = get_trace_id()
    try:
        # Define tool names registered in FastMCP servers
        tools = ["get_product_info", "web_search"]
        await asyncio.sleep(0)  # cooperative yield for async context
        LOGGER.info("MCP tool registry initialized", tools=tools, trace_id=trace_id)
        return tools
    except Exception as e:
        LOGGER.error("Failed to initialize MCP tool registry", trace_id=trace_id, error=str(e))
        return []


# ======================================================================
# Event Loop Safe Scheduler
# ======================================================================
def schedule_mcp_tool_loading(target_obj=None):
    """
    Schedule MCP tool registry loading safely within an async-aware context.

    Behavior:
        - Detects if event loop is running (e.g., FastAPI server context).
        - If running → schedules as background coroutine.
        - If not running → executes synchronously until completion.
        - Optionally binds tool names to target object (e.g., AgenticRAG).

    Args:
        target_obj (object, optional): Instance to attach the `mcp_tools` list.

    Returns:
        None
    """
    trace_id = get_trace_id()
    try:
        loop = asyncio.get_event_loop()

        async def _load_and_store():
            """Internal coroutine that initializes and assigns tool registry."""
            try:
                tools = await load_mcp_tools_async()
                if target_obj is not None:
                    target_obj.mcp_tools = tools
                    LOGGER.info("MCP tools bound to target object", target=type(target_obj).__name__, tool_count=len(tools), trace_id=trace_id)
            except Exception as inner_err:
                LOGGER.error("MCP async load/store routine failed", trace_id=trace_id, error=str(inner_err))

        if loop.is_running():
            loop.create_task(_load_and_store())
            LOGGER.info("MCP tool loader scheduled as background task", trace_id=trace_id)
        else:
            loop.run_until_complete(_load_and_store())
            LOGGER.info("MCP tool loader executed synchronously", trace_id=trace_id)

    except Exception as e:
        LOGGER.error("Failed to start MCP tool loader", trace_id=trace_id, error=str(e))
