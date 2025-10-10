# prod_assistant/utils/mcp_tool_loader.py
"""
================================================================================
 MCP Tool Loader Utilities
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : utils.mcp_tool_loader
- Version     : 1.0.0
- Created on  : 2025-10-10
- Environment : Python 3.11.13 | LangGraph | MCP Adapter | StructLog
================================================================================

Description
-----------
Provides asynchronous utilities for loading MCP (Multi-Channel Processing)
tools safely and efficiently, with structured observability and event-loop
awareness.

Architecture Context
--------------------
Layer:        Integration / MCP Adapter
Upstream:     AgenticRAG Orchestrator
Downstream:   External MCP Servers (product_search_server, etc.)

Key Responsibilities
--------------------
• Load MCP tool metadata asynchronously.
• Handle runtime event-loop conflicts gracefully.
• Provide detailed structured logs with per-trace correlation.

Engineering Standards
---------------------
• FAANGM-grade error handling (never crash on async runtime mismatch).
• Semantic structured logging with `trace_id`.
• Strict behavior-oriented docstrings for maintainability.
"""

import asyncio
from prod_assistant.core.globals import LOGGER
from prod_assistant.core.trace import get_trace_id


# ======================================================================
# Async MCP Tool Loader
# ======================================================================
async def load_mcp_tools_async(mcp_client):
    """
    Load available MCP tools asynchronously with robust error handling.

    Behavior
    --------
    - Uses the provided `mcp_client` to fetch all available tools.
    - Extracts tool names and logs them under the current `trace_id`.
    - Returns an empty list if the retrieval fails, ensuring the
      pipeline remains non-blocking.

    Parameters
    ----------
    mcp_client : object
        The initialized MCP client instance exposing an async `get_tools()` method.

    Returns
    -------
    list
        A list of MCP tool metadata objects if successful, else an empty list.

    Raises
    ------
    None
        All exceptions are caught and logged to prevent runtime interruption.
    """
    trace_id = get_trace_id()
    try:
        tools = await mcp_client.get_tools()
        tool_names = [t.name for t in tools]
        LOGGER.info(
            "MCP tools loaded successfully",
            tools=tool_names,
            trace_id=trace_id,
        )
        return tools
    except Exception as e:
        LOGGER.error(
            "Failed to load MCP tools asynchronously",
            trace_id=trace_id,
            error=str(e),
        )
        return []


# ======================================================================
# Event Loop Safe Scheduler
# ======================================================================
def schedule_mcp_tool_loading(mcp_client):
    """
    Schedule MCP tool loading in an event-loop-safe manner.

    Behavior
    --------
    - Detects whether an asyncio event loop is already running.
    - If not running → runs `load_mcp_tools_async()` synchronously.
    - If already running → schedules a coroutine task to avoid conflicts.
    - Ensures no unhandled runtime warnings or blocking in async contexts.

    Parameters
    ----------
    mcp_client : object
        The MCP client instance whose tools need to be loaded.

    Returns
    -------
    None
        Executes asynchronously or synchronously depending on loop state.

    Raises
    ------
    None
        All exceptions are caught and logged with trace context.
    """
    trace_id = get_trace_id()
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(load_mcp_tools_async(mcp_client))
        else:
            loop.run_until_complete(load_mcp_tools_async(mcp_client))
    except Exception as e:
        LOGGER.error(
            "Failed to start MCP tool loader",
            trace_id=trace_id,
            error=str(e),
        )
