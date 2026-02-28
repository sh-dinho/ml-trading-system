import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.paths import config_path
from src.utils.logger import get_logger

logger = get_logger("config_loader")


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML parsing error in {path}: {e}")


def load_config(
    filename: str,
    env: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Args:
        filename: Name of YAML file inside config/
        env: Optional environment suffix (e.g., 'dev', 'prod')
             Loads model.yaml + model.dev.yaml (if exists)
        overrides: Dict of runtime overrides

    Returns:
        Dict configuration
    """

    base_path = config_path(filename)
    config = _read_yaml(base_path)

    # Environment-specific config
    if env:
        env_file = filename.replace(".yaml", f".{env}.yaml")
        env_path = config_path(env_file)
        if env_path.exists():
            logger.info(f"Loading environment config: {env_file}")
            env_config = _read_yaml(env_path)
            config.update(env_config)

    # Apply overrides
    if overrides:
        logger.info("Applying runtime config overrides")
        config.update(overrides)

    logger.info(f"Loaded config: {filename}")
    return config