# prod_assistant/router/main.py

"""
==========================================================
 PulseFlow Router Module
 ----------------------------------------------------------
 File: prod_assistant/router/main.py

 Description:
     Core FastAPI entrypoint for the PulseFlow backend service.
     This module defines:
         • HTTP routing for the chat and observability endpoints
         • Request → Agentic RAG workflow orchestration
         • Global exception handling and trace correlation
         • LangSmith-integrated tracing wrapper for full observability

 Architecture Context:
     ├── Layer:  User / API Gateway
     ├── Upstream: Chat UI (HTML Form, /get route)
     ├── Downstream: AgenticRAG workflow → LangSmith → CloudWatch
     ├── Observability: Trace IDs via LangSmith + Structured JSON logs
     └── Deployment: AWS ECS / EKS (FastAPI container service)

 Key Features:
     ✅ Structured JSON logging with per-request trace_id
     ✅ Unified observability endpoints (/info, /health, /metrics)
     ✅ LangSmith trace wrapper (compatible ≤ v0.4.32)
     ✅ Graceful exception management using ProductAssistantException
     ✅ Cloud-native readiness for S3 log archival and monitoring

 Version:     v1.1.0 – Observability Milestone
 Author:      Bhagwat Chate
 Organization: iDataflow.ai
 Date:        2025-10-05
 ----------------------------------------------------------
 Engineering Standards Followed:
     • FAANGM-grade structure and docstrings
     • Modular exception safety with explicit error boundaries
     • Semantic logging (event, message, trace_id)
     • Compatibility with LangSmith SDK ≤ 0.4.32
     • Production-ready FastAPI lifecycle hooks and middleware
==========================================================
"""

import os
import platform
import warnings
from datetime import datetime

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --- Suppress known LangSmith warnings ---
warnings.filterwarnings("ignore", message="No add_extra helper")

# --- Core bootstrap ---
from prod_assistant.core.bootstrap import bootstrap_app
from prod_assistant.core.globals import get_config, LOGGER
from prod_assistant.core.trace import new_trace_id, get_trace_id
from prod_assistant.exception.custom_exception import ProductAssistantException

# Initialize configuration + logger early
bootstrap_app()

# --- LangSmith integration ---
from langsmith import traceable, run_helpers

# --- Agentic workflow ---
from prod_assistant.workflow.agentic_workflow_with_mcp_websearch import AgenticRAG


# ==========================================================
# LangSmith Top-Level Trace Wrapper
# ==========================================================
@traceable(name="PulseFlowRequest")
def run_pulseflow_agent(query: str) -> str:
    """Execute a LangSmith-traced PulseFlow request."""
    trace_id = get_trace_id()
    try:
        if hasattr(run_helpers, "add_extra"):
            run_helpers.add_extra({"trace_id": trace_id})
        elif hasattr(run_helpers, "set_global_extra"):
            run_helpers.set_global_extra({"trace_id": trace_id})
        else:
            print(f"[LangSmith] No add_extra helper in this version, trace_id={trace_id}")
    except Exception as e:
        print(f"[LangSmith metadata injection skipped] {e}")

    try:
        return AgenticRAG().run(query)
    except Exception as e:
        # Wrap any runtime errors from the agent
        raise ProductAssistantException(e)


# ==========================================================
# FastAPI Initialization
# ==========================================================
app = FastAPI(title="PulseFlow", version="1.0.0")

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Record app startup time
start_time = datetime.utcnow()


# ==========================================================
# Observability Endpoints
# ==========================================================
@app.get("/info")
async def system_info() -> dict:
    """Return runtime metadata for observability systems."""
    return {
        "app": "PulseFlow",
        "version": "1.0.0",
        "environment": os.getenv("PULSEFLOW_APP_ENV", "DEV"),
        "python_version": platform.python_version(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/health")
async def health() -> dict:
    """Lightweight liveness probe endpoint."""
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> dict:
    """Expose basic runtime metrics."""
    uptime = (datetime.utcnow() - start_time).total_seconds()
    return {
        "uptime_seconds": int(uptime),
        "active_traces": "N/A",
    }


# ==========================================================
# Chat and Root Endpoints
# ==========================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render the PulseFlow chat UI."""
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/get", response_class=HTMLResponse)
async def chat(msg: str = Form(...)) -> str:
    """Process chat messages via the Agentic RAG workflow."""
    try:
        LOGGER.info("Received chat request", message=msg)
        answer = run_pulseflow_agent(msg)
        LOGGER.info("Agentic RAG response generated", response=answer[:200])
        return answer
    except ProductAssistantException as pe:
        # Custom exception already enriched with traceback
        LOGGER.error("Agentic RAG failed", error=str(pe))
        return (
            "⚠️ Internal error while processing your request. "
            "Please retry or contact support with the trace ID."
        )
    except Exception as e:
        # Wrap any unexpected errors
        wrapped_exc = ProductAssistantException(e)
        LOGGER.error("Unhandled exception in chat route", error=str(wrapped_exc))
        return (
            "⚠️ Unexpected system issue. Please try again later. "
            "Our team has been notified."
        )


# ==========================================================
# Global Exception Handlers
# ==========================================================
@app.exception_handler(ProductAssistantException)
async def product_assistant_exception_handler(request: Request, exc: ProductAssistantException):
    """
    Global handler for ProductAssistantException.
    Returns structured JSON for API clients (non-HTML consumers).
    """
    LOGGER.error("Global ProductAssistantException caught", error=str(exc))
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": exc.error_message,
            "file": exc.file_name,
            "line": exc.lineno,
            "trace_id": request.headers.get("X-Trace-ID", "unknown"),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all handler for unexpected exceptions."""
    wrapped_exc = ProductAssistantException(exc)
    LOGGER.critical("Unhandled system exception", error=str(wrapped_exc))
    return JSONResponse(
        status_code=500,
        content={
            "error": "Unexpected Error",
            "message": "An unexpected error occurred.",
            "trace_id": request.headers.get("X-Trace-ID", "unknown"),
        },
    )


# ==========================================================
# Trace Middleware (HTTP Layer)
# ==========================================================
@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    """Attach a ``trace_id`` to every incoming request and response."""
    trace_id = new_trace_id()
    LOGGER.info("new request trace initialized", trace_id=trace_id, path=request.url.path)
    response = await call_next(request)
    response.headers["X-Trace-ID"] = trace_id
    return response


# ==========================================================
# Local Debug Entrypoint
# ==========================================================
if __name__ == "__main__":
    """Run the FastAPI app locally for development/debugging."""
    try:
        cfg = get_config()
        port = cfg["app"].get("port", 8080)
        LOGGER.info("PulseFlow app is running", port=port)
        uvicorn.run(app, host="0.0.0.0", port=port)
    except Exception as e:
        wrapped_exc = ProductAssistantException(e)
        LOGGER.critical("Fatal error during PulseFlow startup", error=str(wrapped_exc))
        raise wrapped_exc
