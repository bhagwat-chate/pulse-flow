# prod_assistant/router/main.py

"""
================================================================================
 PulseFlow API Entrypoint
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : router.main
- Version     : 1.2.0  |  Modular Router Refactor
- Created on  : 2025-10-10
================================================================================

Description
-----------
Central FastAPI initialization module that bootstraps the PulseFlow
application, mounts modular routers, registers middlewares, and sets
global lifecycle hooks for observability and graceful shutdown.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# --- Core Bootstrap ---
from prod_assistant.core.bootstrap import bootstrap_app
from prod_assistant.core.globals import get_config, LOGGER
from prod_assistant.exception.custom_exception import ProductAssistantException

# --- Load configuration and logger early ---
bootstrap_app()

# --- Router Imports ---
from prod_assistant.router.chat_router import router as chat_router
from prod_assistant.router.system_router import router as system_router
from prod_assistant.router.exception_handlers import register_exception_handlers
from prod_assistant.router.trace_middleware import add_trace_middleware


# ======================================================================
# FastAPI Initialization
# ======================================================================
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

# Register Routers
app.include_router(chat_router, prefix="", tags=["Chat"])
app.include_router(system_router, prefix="/system", tags=["System"])

# Register Middleware + Exceptions
add_trace_middleware(app)
register_exception_handlers(app)


# ======================================================================
# Lifecycle Hooks
# ======================================================================
@app.on_event("shutdown")
async def on_shutdown():
    """Log graceful application shutdown."""
    LOGGER.info("PulseFlow shutting down gracefully", trace_id="system")


# ======================================================================
# Local Development Entrypoint
# ======================================================================
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
