"""
================================================================================
 PulseFlow Application Bootstrap Module
================================================================================
- Author      : Bhagwat Chate
- Project     : PulseFlow – Multi-Agent Product Intelligence System
- Module      : core.bootstrap
- Version     : 1.0.0
- Created on  : 2025-10-07
- Last Updated: 2025-10-10
- Environment : Python 3.11.13 | FastAPI | LangGraph | AWS Secrets Manager | StructLog
================================================================================

Overview
--------
This module initializes the **PulseFlow** runtime environment before any
workflow, agent, or FastAPI component starts. It is responsible for
loading YAML configurations, injecting environment secrets, and setting up
structured logging for full observability. The resulting configuration and
logger are stored in global registries and used by all downstream components.

Core Responsibilities
---------------------
- Load and merge configuration files (`config_base.yaml` + environment file).
- Resolve `${VAR}` placeholders from `.env` (DEV) or AWS Secrets (PROD).
- Initialize the `CustomLogger` instance and attach it globally.
- Maintain consistent environment context across all PulseFlow modules.
- Print system initialization logs for traceability.

Workflow Topology
-----------------
main.py / FastAPI Entrypoint
    ↓
bootstrap_app()
    ↓
load .env or AWS Secrets
    ↓
merge base + environment YAML
    ↓
register global CONFIG + LOGGER
    ↓
PulseFlow runtime initialized successfully

External Integrations
---------------------
- **dotenv** — Loads `.env` for local environment variables.
- **AWS Secrets Manager** — Fetches production secrets dynamically.
- **StructLog** — Handles structured JSON-based logging.
- **Globals Registry** — Maintains shared state for configuration and logger.

Changelog
---------
v1.0.0 (2025-10-07)
    • Implemented configuration bootstrap pipeline.
    • Added environment variable resolution and YAML merging.
    • Integrated AWS Secrets Manager and structured logging system.
    • Globalized configuration and logger for the runtime environment.

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
    """
    Load a YAML configuration file and replace all `${VAR}` placeholders
    with corresponding environment variable values.

    Parameters
    ----------
    path : str
        Absolute or relative path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed YAML content as a Python dictionary with placeholders replaced.
        Returns an empty dictionary `{}` if the file cannot be read or parsed.

    Example
    -------
    If `config.yaml` contains:
        api_key: ${OPENAI_API_KEY}

    and environment has:
        OPENAI_API_KEY="sk-xyz123"

    then the function will load:
        {'api_key': 'sk-xyz123'}
    """
    try:
        # Ensure the file exists before reading
        if not Path(path).exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        # Read YAML file content
        text = Path(path).read_text(encoding="utf-8")

        # Replace ${VAR} placeholders with environment values
        pattern = re.compile(r"\$\{([^}]+)\}")
        for var in pattern.findall(text):
            val = os.getenv(var, f"<missing:{var}>")
            text = text.replace(f"${{{var}}}", val)

        # Safely parse YAML content
        parsed_yaml = yaml.safe_load(text)
        return parsed_yaml or {}

    except FileNotFoundError as fnf:
        print(f"[bootstrap:load_yaml_with_env] File not found: {fnf}")
        return {}

    except yaml.YAMLError as ye:
        print(f"[bootstrap:load_yaml_with_env] YAML parsing error in {path}: {ye}")
        return {}

    except Exception as e:
        print(f"[bootstrap:load_yaml_with_env] Unexpected error while loading {path}: {e}")
        return {}


def merge_dicts(base: dict, override: dict) -> dict:
    """
    Recursively merge two dictionaries where the `override` values
    take precedence over the `base` configuration.

    This function is primarily used during PulseFlow bootstrap to merge
    environment-specific configurations (e.g., `config_dev.yaml`) with
    the base configuration (`config_base.yaml`).

    Parameters
    ----------
    base : dict
        The base configuration dictionary (for example, base YAML config).
    override : dict
        The environment-specific configuration dictionary.

    Returns
    -------
    dict
        A merged configuration dictionary with `override` values
        taking precedence over `base`.

    Example
    -------
    # >>> base = {'app': {'port': 8080}, 'llm': {'provider': 'openai'}}
    # >>> override = {'app': {'port': 9090}}
    # >>> merge_dicts(base, override)
    {'app': {'port': 9090}, 'llm': {'provider': 'openai'}}
    """
    try:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge_dicts(base[key], value)
            else:
                base[key] = value
        return base

    except Exception as e:
        # Log any merge issues without interrupting bootstrap
        if hasattr(globals, "LOGGER") and globals.LOGGER:
            globals.LOGGER.warning("Configuration merge failed", error=str(e))
        else:
            print(f"[merge_dicts] Configuration merge failed: {e}")
        return base


def bootstrap_app() -> None:
    """
    Initialize the **PulseFlow** runtime by performing the following:
    1. Detects environment (`dev` or `prod`).
    2. Loads configuration files and secrets.
    3. Merges base + environment-specific YAML configurations.
    4. Initializes global logger and registers configuration globally.

    Workflow Steps
    ---------------
    - If `APP_ENV=prod`, connect to AWS Secrets Manager to fetch credentials.
    - If `APP_ENV=dev`, load `.env` file for local testing.
    - Resolve all `${VAR}` placeholders recursively in the YAML files.
    - Register configuration and logger inside `prod_assistant.core.globals`.

    Raises
    ------
    botocore.exceptions.ClientError
        If AWS Secrets Manager retrieval fails.
    FileNotFoundError
        If YAML configuration files are missing.

    Logs
    ----
    Logs environment, config source path, and version during initialization.

    Example
    -------
    >>> bootstrap_app()
    {"event": "Environment: DEV"}
    {"event": "Loaded configuration from prod_assistant/config/config_dev.yaml"}
    {"event": "PulseFlow v1.0.0 bootstrapped successfully"}
    """
    try:
        env = os.getenv("APP_ENV", "dev").lower()

        # Load secrets
        if env == "prod":
            client = boto3.client("secretsmanager", region_name=os.getenv("AWS_REGION", "us-east-1"))
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

    except FileNotFoundError as fnf:
        print(f"[bootstrap:bootstrap_app] Missing configuration file: {fnf}")

    except yaml.YAMLError as ye:
        print(f"[bootstrap:bootstrap_app] YAML parsing error: {ye}")

    except Exception as e:
        print(f"[bootstrap:bootstrap_app] Unexpected bootstrap error: {e}")


# ---------------------------------------------------------------------
# CLI run for debug
# ---------------------------------------------------------------------
if __name__ == "__main__":
    """
    Debug entrypoint for standalone execution of the bootstrap process.

    This allows developers to verify YAML merging, secret substitution,
    and logger initialization locally without launching the entire FastAPI app.

    Example
    -------
    $ python -m prod_assistant.core.bootstrap
    """
    bootstrap_app()
