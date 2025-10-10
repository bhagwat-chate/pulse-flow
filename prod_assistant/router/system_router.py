# prod_assistant/router/system_router.py
"""
System Router
=============
Provides runtime observability endpoints:
- `/system/info` → Metadata
- `/system/health` → Liveness probe
- `/system/metrics` → Uptime + metrics
"""

import os
import platform
from datetime import datetime
from fastapi import APIRouter

router = APIRouter()
start_time = datetime.utcnow()


@router.get("/info")
async def system_info():
    """Return runtime and environment metadata."""
    return {
        "app": "PulseFlow",
        "version": "1.0.0",
        "environment": os.getenv("PULSEFLOW_APP_ENV", "DEV"),
        "python_version": platform.python_version(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/health")
async def health():
    """Liveness probe endpoint."""
    return {"status": "ok"}


@router.get("/metrics")
async def metrics():
    """Expose runtime metrics like uptime."""
    uptime = (datetime.utcnow() - start_time).total_seconds()
    return {"uptime_seconds": int(uptime), "active_traces": "N/A"}
