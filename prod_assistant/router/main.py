# prod_assistant/router/main.py

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
from prod_assistant.core import globals
from prod_assistant.core.globals import get_config
bootstrap_app()  # Initializes config + logger
from prod_assistant.core.globals import LOGGER

# Safe to import downstream modules now
from prod_assistant.workflow.agentic_workflow_with_mcp_websearch import AgenticRAG

app = FastAPI(title="PulseFlow - E-commerce Product Intelligence")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- FastAPI Endpoints ----------
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/get", response_class=HTMLResponse)
async def chat(msg: str = Form(...)):
    """Call the Agentic RAG workflow."""
    LOGGER.info("Received chat request", message=msg)

    rag_agent = AgenticRAG()
    answer = rag_agent.run(msg)   # run() already returns final answer string
    LOGGER.info("Agentic RAG response generated", response=answer[:200])

    return answer


@app.middleware("http")
async def add_trace_id(request: Request, call_next):
    """Middleware to attach a new trace_id to each incoming request."""
    trace_id = new_trace_id()
    LOGGER.info("🪪 New request trace initialized", trace_id=trace_id, path=request.url.path)

    # Attach trace_id to response headers as well (optional, for debugging)
    response = await call_next(request)
    response.headers["X-Trace-ID"] = trace_id
    return response


# ---------- Local Debug Entrypoint ----------
if __name__ == "__main__":
    cfg = get_config()
    port = cfg["app"].get("port", 8080)
    LOGGER.info("PulseFlow app is running", port=port)
    uvicorn.run(app, host="0.0.0.0", port=port)
