# prod_assistant/core/bootstrap.py

"""
Application Bootstrap Module
=============================

This module is responsible for initializing the entire PulseFlow runtime
environment before any other component executes.

It performs the following critical tasks:
------------------------------------------------
1. **Configuration Loading**
   - Reads the base YAML configuration (`config_base.yaml`) shared across all environments.
   - Merges it with environment-specific overrides (`config_dev.yaml`, `config_prod.yaml`).
   - Dynamically replaces placeholders (e.g., `${OPENAI_API_KEY}`) with actual environment values.

2. **Secret Management**
   - In **DEV/LOCAL**, loads secrets from the `.env` file.
   - In **PROD**, securely fetches secrets from **AWS Secrets Manager** using the provided
     `AWS_REGION` and `AWS_SECRET_NAME`.

3. **Logger Initialization**
   - Creates a structured JSON logger (via `structlog`), configured for both
     console and file output.
   - Registers the logger globally so it can be reused by all modules.

4. **Global Registry Setup**
   - Stores the merged configuration and initialized logger into
     `prod_assistant.core.globals` to ensure single-instance access across the app.

5. **Version Traceability**
   - Logs the active environment and application version (`__version__`) for
     audit and observability purposes.

This module is invoked once during application startup, typically from
`main.py` or the FastAPI entrypoint, using:

    from prod_assistant.core.bootstrap import bootstrap_app
    bootstrap_app()

Author: Bhagwat Chate
Project: PulseFlow (E-Commerce Product Intelligence)
Version: 1.0.0
"""


import os
import re
import yaml
import json
import boto3
from pathlib import Path
from dotenv import load_dotenv
from prod_assistant.core import globals
from prod_assistant.core.logger import CustomLogger
from versions import __version__


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------
def load_yaml_with_env(path: str) -> dict:
    """Load YAML and replace ${VAR} placeholders with environment values."""
    pattern = re.compile(r"\$\{([^}]+)\}")
    text = Path(path).read_text()
    for var in pattern.findall(text):
        val = os.getenv(var, f"<missing:{var}>")
        text = text.replace(f"${{{var}}}", val)
    return yaml.safe_load(text) or {}


def merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge two dictionaries (override takes precedence)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            merge_dicts(base[k], v)
        else:
            base[k] = v
    return base


# ---------------------------------------------------------------------
# Main bootstrap
# ---------------------------------------------------------------------
def bootstrap_app() -> None:
    """Initialize configuration, secrets, and global logger."""
    env = os.getenv("APP_ENV", "dev").lower()

    # Load secrets
    if env == "prod":
        client = boto3.client(
            "secretsmanager",
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )
        secret_name = os.getenv("AWS_SECRET_NAME")
        response = client.get_secret_value(SecretId=secret_name)
        secrets = json.loads(response["SecretString"])
        os.environ.update(secrets)
        env_cfg_path = "prod_assistant/config/config_prod.yaml"
    else:
        load_dotenv()
        env_cfg_path = "prod_assistant/config/config_dev.yaml"

    # Merge base + environment config
    base_cfg = load_yaml_with_env("prod_assistant/config/config_base.yaml")
    env_cfg = load_yaml_with_env(env_cfg_path)
    final_cfg = merge_dicts(base_cfg, env_cfg)
    globals.set_config(final_cfg)

    # Initialize and register logger
    logger = CustomLogger().get_logger(__file__)
    globals.LOGGER = logger

    # Log summary
    logger.info(f"Environment: {env.upper()}")
    logger.info(f"Loaded configuration from {env_cfg_path}")
    logger.info(f"PulseFlow v{__version__} bootstrapped successfully")


# ---------------------------------------------------------------------
# CLI run for debug
# ---------------------------------------------------------------------
if __name__ == "__main__":
    bootstrap_app()
