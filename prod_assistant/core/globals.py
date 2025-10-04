# prod_assistant/core/globals.py

"""
Central registry for app-wide singletons like config, logger, and cache.
Ensures they are initialized once and reused everywhere.
"""

from typing import Optional, Any

CONFIG: Optional[Any] = None
LOGGER: Optional[Any] = None
CACHE: Optional[Any] = None


def set_config(cfg):  # called once at bootstrap
    global CONFIG
    CONFIG = cfg


def get_config():
    if CONFIG is None:
        raise RuntimeError("Config accessed before initialization. Call bootstrap_app() first.")

    return CONFIG
