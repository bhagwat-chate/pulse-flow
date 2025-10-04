# # main.py
#
# import sys
# from prod_assistant.core.bootstrap import bootstrap_app
# from prod_assistant.router.main import
#
# def main():
#     """Entry point for the pulse-flow CLI."""
#     print("🚀 Pulse-Flow is running!")
#
#     bootstrap_app()
#
#
# if __name__ == "__main__":
#     main()


# main.py
"""
Main Entrypoint for PulseFlow
=============================

This script acts as the top-level launcher for the entire PulseFlow application.

Responsibilities:
-----------------
1. Bootstrap the environment (loads .env / AWS secrets / config / logger).
2. Import and start the FastAPI app (from prod_assistant.router.main).
3. Serve as a unified execution entry — for both local dev and container runs.

Usage:
------
python main.py
"""

import uvicorn
from prod_assistant.core.bootstrap import bootstrap_app
from prod_assistant.core import globals


def main():
    """Top-level entrypoint for PulseFlow runtime."""
    print("🚀 PulseFlow is starting up...")

    # 1️⃣ Initialize configuration, secrets, and logger
    bootstrap_app()
    LOGGER = globals.LOGGER
    CONFIG = globals.get_config()

    # 2️⃣ Import FastAPI app (only after bootstrap)
    from prod_assistant.router.main import app

    # 3️⃣ Start the server
    port = CONFIG["app"].get("port", 8080)
    LOGGER.info("PulseFlow app is running", port=port)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()

