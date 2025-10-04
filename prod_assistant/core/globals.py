# prod_assistant/core/globals.py

"""
Central registry for app-wise singletons like config, cache, logger.
Ensure they are initialized once and reused everywhere.
"""

from typing import Optional, Any

CONFIG: Optional[Any] = None
LOGGER: Optional[Any] = None
CACHE: Optional[Any] = None


def set_config(cng):
    global CONFIG
    CONFIG = cng


def get_config():
    if CONFIG is None:
        raise RuntimeError("Config accessed before initialization. Call bootstrap_app() first.")

    return CONFIG
