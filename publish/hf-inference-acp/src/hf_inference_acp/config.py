"""Configuration file handling for hf-inference-acp."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path.home() / ".config" / "hf-inference"
CONFIG_FILE = CONFIG_DIR / "hf.config.yaml"

DEFAULT_MODEL = "hf.moonshotai/Kimi-K2-Instruct-0905"


def get_hf_token() -> str | None:
    """Get HF_TOKEN from environment."""
    return os.environ.get("HF_TOKEN")


def has_hf_token() -> bool:
    """Check if HF_TOKEN is present in environment."""
    return get_hf_token() is not None


def ensure_config_exists() -> Path:
    """Ensure config directory and file exist, creating from template if needed.

    Returns:
        Path to the config file
    """
    from importlib.resources import files

    # Create directory if it doesn't exist
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Create config file from template if it doesn't exist
    if not CONFIG_FILE.exists():
        resource_path = (
            files("hf_inference_acp").joinpath("resources").joinpath("hf.config.yaml")
        )
        if resource_path.is_file():
            template_content = resource_path.read_text()
            CONFIG_FILE.write_text(template_content)

    return CONFIG_FILE


def load_config() -> dict[str, Any]:
    """Load configuration from the config file.

    Returns:
        Configuration dictionary
    """
    config_path = ensure_config_exists()

    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}

    return {}


def update_model_in_config(model: str) -> None:
    """Update the default_model in the config file.

    Args:
        model: The model name to set as default
    """
    config_path = ensure_config_exists()
    config = load_config()
    config["default_model"] = model

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_default_model() -> str:
    """Get the default model from config, or return the default.

    Returns:
        The default model name
    """
    config = load_config()
    return config.get("default_model", DEFAULT_MODEL)
