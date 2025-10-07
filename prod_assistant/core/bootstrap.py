# prod_assistant/core/bootstrap.py

"""
================================================================================
 PulseFlow Application Bootstrap Module
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : core.bootstrap
- Version     : 1.0.0
- Created on  : 2025-10-07
- Last Updated: 2025-10-07
- Environment : Python 3.11.13 | FastAPI | LangGraph | AWS Secrets Manager | StructLog
================================================================================

This module bootstraps the entire **PulseFlow** runtime environment by loading
configuration files, resolving environment variables, managing secrets, and
initializing global components such as structured logging and configuration
registries. It is the first entrypoint executed before any workflow, agent, or
server component runs.

Core Responsibilities
---------------------
- Load base and environment-specific YAML configurations.
- Replace `${VAR}` placeholders with values from `.env` (DEV) or AWS Secrets Manager (PROD).
- Merge configurations into a single global object stored in `core.globals`.
- Initialize a structured `CustomLogger` instance and register it globally.
- Provide environment, version, and configuration traceability for observability.

Workflow Topology
-----------------
    main.py / FastAPI → bootstrap_app()
        → load .env or AWS Secrets
        → merge base + env YAML
        → set globals.CONFIG and globals.LOGGER
        → application ready

External Integrations
---------------------
- **dotenv** — Loads local environment variables in development mode.
- **AWS Secrets Manager** — Securely retrieves production secrets.
- **StructLog** — Provides structured, JSON-based logging for all modules.
- **Globals Registry** — Stores unified configuration and logger instances.

Changelog
---------
v1.0.0  (2025-10-07)
    • Initial stable bootstrap implementation for environment-aware configuration.
    • Added YAML placeholder substitution and recursive merge strategy.
    • Integrated AWS Secrets Manager for secure production deployments.
    • Registered structured JSON logger globally via `CustomLogger`.

License
-------
Copyright © 2025 Bhagwat Chate.
This code is part of the **PulseFlow** system under the personal projects umbrella.
All rights reserved.
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
