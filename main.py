# main.py
"""
================================================================================
 PulseFlow — Main Entrypoint
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : main
- Version     : 1.0.0
- Created on  : 2025-10-10
- Environment : Python 3.11.13 | FastAPI | LangGraph | StructLog
================================================================================

Description
-----------
This script serves as the **top-level launcher** for the entire PulseFlow
application. It initializes environment configuration, loads all secrets,
sets up structured logging, and launches the FastAPI web server.

Architecture Context
--------------------
Layer:        Application Entrypoint
Upstream:     Local CLI or Container CMD
Downstream:   FastAPI Router → AgenticRAG Workflow → LangSmith

Core Responsibilities
---------------------
1. Bootstrap the environment (.env / AWS Secrets / Config / Logger).
2. Import and start the FastAPI app from `prod_assistant.router.main`.
3. Serve as a unified launcher for local development and container execution.

Engineering Standards
---------------------
• FAANGM-grade structured startup sequence.
• Explicit bootstrap and configuration validation.
• Clean modular separation between init and runtime layers.
• Semantic structured logging (event, port, trace_id).
• Fully container-compatible for AWS ECS / EKS deployment.
"""

import uvicorn
from prod_assistant.core.bootstrap import bootstrap_app
from prod_assistant.core import globals
from prod_assistant.exception.custom_exception import ProductAssistantException


def main():
    """
    Top-level runtime entrypoint for PulseFlow.

    Behavior
    --------
    - Initializes the global app environment (config, secrets, logging).
    - Imports the FastAPI router only after bootstrap completes.
    - Launches the FastAPI server using Uvicorn on configured port.
    - Handles initialization errors gracefully using ProductAssistantException.

    Returns
    -------
    None
        Executes the server runtime.
    """
    print("PulseFlow is starting up...")

    try:
        # ------------------------------------------------------------------
        # 1️⃣ Initialize configuration, secrets, and structured logger
        # ------------------------------------------------------------------
        bootstrap_app()
        LOGGER = globals.LOGGER
        CONFIG = globals.get_config()

        LOGGER.info(
            "Environment bootstrapped successfully",
            app=CONFIG["app"]["name"],
            env=CONFIG["app"].get("env", "base"),
        )

        # ------------------------------------------------------------------
        # 2️⃣ Import FastAPI app (after bootstrap)
        # ------------------------------------------------------------------
        from prod_assistant.router.main import app

        # ------------------------------------------------------------------
        # 3️⃣ Start Uvicorn FastAPI server
        # ------------------------------------------------------------------
        port = CONFIG["app"].get("port", 8080)
        LOGGER.info("PulseFlow FastAPI service started...", port=port)
        uvicorn.run(app, host="0.0.0.0", port=port)

    except Exception as e:
        # Catch bootstrap or startup-level failures
        wrapped_exc = ProductAssistantException(e)
        print(f"Fatal startup error: {wrapped_exc}")
        if 'LOGGER' in locals():
            LOGGER.critical("PulseFlow startup failed", error=str(wrapped_exc))
        raise wrapped_exc


if __name__ == "__main__":
    main()
