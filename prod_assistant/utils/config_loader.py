# prod_assistant/utils/config_loader.py

import os
import yaml
from pathlib import Path
from typing import Optional


def load_config(config_path: Optional[str] = None) -> dict:
    """
    Load YAML configuration file with robust path resolution.
    Priority:
      1. CONFIG_PATH environment variable
      2. Explicit function argument
      3. Default: <project_root>/config/config.yaml
    """
    # resolve config path priority
    config_path = os.getenv("CONFIG_PATH") or config_path or str(_project_root() / "config" / "config.yaml")

    path = Path(config_path)
    if not path.is_absolute():
        path = _project_root() / path

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]
