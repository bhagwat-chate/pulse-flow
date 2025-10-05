# # prod_assistant/router/main.py
#
# import uvicorn
# from fastapi import FastAPI, Request, Form
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from prod_assistant.core.trace import new_trace_id, get_trace_id
# from fastapi import Request
#
# # --- Bootstrap early ---
# from prod_assistant.core.bootstrap import bootstrap_app
# from prod_assistant.core.globals import get_config
# bootstrap_app()  # Initializes config + logger
# from prod_assistant.core.globals import LOGGER
#
# # Safe to import downstream modules now
# from prod_assistant.workflow.agentic_workflow_with_mcp_websearch import AgenticRAG
#
# app = FastAPI(title="PulseFlow - E-commerce Product Intelligence")
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")
#
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
#
#
# # ---------- FastAPI Endpoints ----------
# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     return templates.TemplateResponse("chat.html", {"request": request})
#
#
# @app.post("/get", response_class=HTMLResponse)
# async def chat(msg: str = Form(...)):
#     """Call the Agentic RAG workflow."""
#     LOGGER.info("Received chat request", message=msg)
#
#     rag_agent = AgenticRAG()
#     answer = rag_agent.run(msg)   # run() already returns final answer string
#
#     LOGGER.info("Agentic RAG response generated", response=answer[:200])
#
#     return answer
#
#
# @app.middleware("http")
# async def add_trace_id(request: Request, call_next):
#     """Middleware to attach a new trace_id to each incoming request."""
#     trace_id = new_trace_id()
#     LOGGER.info("new request trace initialized", trace_id=trace_id, path=request.url.path)
#
#     # Attach trace_id to response headers as well (optional, for debugging)
#     response = await call_next(request)
#     response.headers["X-Trace-ID"] = trace_id
#     return response
#
#
# # ---------- Local Debug Entrypoint ----------
# if __name__ == "__main__":
#     cfg = get_config()
#     port = cfg["app"].get("port", 8080)
#     LOGGER.info("PulseFlow app is running", port=port)
#     uvicorn.run(app, host="0.0.0.0", port=port)



import warnings
warnings.filterwarnings("ignore", message="No add_extra helper")
import uvicorn
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from prod_assistant.core.trace import new_trace_id, get_trace_id
from fastapi import Request

# --- Bootstrap early ---
from prod_assistant.core.bootstrap import bootstrap_app
from prod_assistant.core.globals import get_config
bootstrap_app()  # Initializes config + logger
from prod_assistant.core.globals import LOGGER
from fastapi import FastAPI
from datetime import datetime
import os

# Record app startup time once
start_time = datetime.utcnow()
# Safe to import downstream modules now
from prod_assistant.workflow.agentic_workflow_with_mcp_websearch import AgenticRAG

# --- LangSmith integration imports ---
from langsmith import traceable, run_helpers



# ==========================================================
# 🔹 LangSmith Top-Level Trace Wrapper
# ==========================================================
from langsmith import traceable
from prod_assistant.core.trace import get_trace_id
from prod_assistant.workflow.agentic_workflow_with_mcp_websearch import AgenticRAG
from langsmith import run_helpers
import platform, datetime, os

@traceable(name="PulseFlowRequest")
def run_pulseflow_agent(query: str):
    """
    Top-level LangSmith traced function ensuring the same trace_id
    flows into LangSmith metadata for correlation.
    Works with LangSmith <=0.4.32 (no add_extra helper).
    """
    trace_id = get_trace_id()

    # ✅ Backward-compatible metadata injection
    try:
        if hasattr(run_helpers, "add_extra"):
            run_helpers.add_extra({"trace_id": trace_id})
        elif hasattr(run_helpers, "set_global_extra"):
            run_helpers.set_global_extra({"trace_id": trace_id})
        else:
            # fallback: log it, so we still see correlation in logs
            print(f"[LangSmith] No add_extra helper in this version, trace_id={trace_id}")
    except Exception as e:
        print(f"[LangSmith metadata injection skipped] {e}")

    rag_agent = AgenticRAG()
    result = rag_agent.run(query)
    return result

# ==========================================================
# 🔹 FastAPI App Initialization
# ==========================================================
app = FastAPI(title="PulseFlow")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================================
# 🔹 Endpoints
# ==========================================================
@app.get("/info")
async def system_info():
    """Simple health/info endpoint for monitoring."""
    return {
        "app": "PulseFlow",
        "version": "1.0.0",
        "environment": os.getenv("APP_ENV", "DEV"),
        "python_version": platform.python_version(),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
    }
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/get", response_class=HTMLResponse)
async def chat(msg: str = Form(...)):
    """Call the Agentic RAG workflow."""
    LOGGER.info("Received chat request", message=msg)

    # Use LangSmith-traced wrapper
    answer = run_pulseflow_agent(msg)

    LOGGER.info("Agentic RAG response generated", response=answer[:200])
    return answer


# ==========================================================
# 🔹 Trace Middleware (HTTP Layer)
# ==========================================================
@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    """Middleware to attach a new trace_id to each incoming request."""
    trace_id = new_trace_id()
    LOGGER.info("new request trace initialized", trace_id=trace_id, path=request.url.path)

    # Attach trace_id to response headers as well (for external correlation)
    response = await call_next(request)
    response.headers["X-Trace-ID"] = trace_id
    return response


# ==========================================================
# 🔹 Local Debug Entrypoint
# ==========================================================
if __name__ == "__main__":
    cfg = get_config()
    port = cfg["app"].get("port", 8080)
    LOGGER.info("PulseFlow app is running", port=port)
    uvicorn.run(app, host="0.0.0.0", port=port)
