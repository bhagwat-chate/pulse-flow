# prod_assistant/router/chat_router.py
"""
Chat Router
===========
Handles chat UI rendering and POST-based user interactions with
the Agentic RAG pipeline, wrapped with LangSmith trace correlation.
"""

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from prod_assistant.core.globals import LOGGER
from prod_assistant.exception.custom_exception import ProductAssistantException
from prod_assistant.workflow.agentic_workflow_with_mcp_websearch import AgenticRAG
from prod_assistant.core.trace import get_trace_id
from langsmith import traceable, run_helpers

templates = Jinja2Templates(directory="templates")
router = APIRouter()


@traceable(name="PulseFlowRequest")
async def run_pulseflow_agent(query: str) -> str:
    """Execute LangSmith-traced PulseFlow query through AgenticRAG pipeline."""
    trace_id = get_trace_id()

    try:
        if hasattr(run_helpers, "add_extra"):
            run_helpers.add_extra({"trace_id": trace_id})
        elif hasattr(run_helpers, "set_global_extra"):
            run_helpers.set_global_extra({"trace_id": trace_id})
        else:
            LOGGER.warning("LangSmith helper unavailable", trace_id=trace_id)
    except Exception as e:
        LOGGER.warning("LangSmith metadata injection failed", trace_id=trace_id, error=str(e))

    try:
        LOGGER.info("Invoking AgenticRAG pipeline", trace_id=trace_id, query=query)
        agent = AgenticRAG()
        result = await agent.run(query, thread_id=trace_id)
        LOGGER.info("AgenticRAG pipeline completed", trace_id=trace_id)
        return result
    except Exception as e:
        LOGGER.error("AgenticRAG pipeline failed", trace_id=trace_id, error=str(e))
        raise ProductAssistantException("AgenticRAG pipeline execution failed", e)


@router.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render chat UI page."""
    return templates.TemplateResponse("chat.html", {"request": request})


@router.post("/get", response_class=HTMLResponse)
async def chat(msg: str = Form(...)) -> str:
    """Handle chat form submission and return AgenticRAG response."""
    try:
        LOGGER.info("Received chat request", message=msg)
        answer = await run_pulseflow_agent(msg)
        LOGGER.info("Response generated successfully", response=answer)
        return answer
    except ProductAssistantException as pe:
        LOGGER.error("Agentic RAG failed", error=str(pe))
        return "Internal error while processing request. Please retry later."
    except Exception as e:
        wrapped_exc = ProductAssistantException(e)
        LOGGER.error("Unhandled exception in chat route", error=str(wrapped_exc))
        return "Unexpected system issue. Please try again later."
