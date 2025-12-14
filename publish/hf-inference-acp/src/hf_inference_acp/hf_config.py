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
    """Get HF token from all available sources.

    Checks in priority order:
    1. Our config file (hf.api_key)
    2. HF_TOKEN environment variable
    3. huggingface_hub token file (~/.cache/huggingface/token)
    """
    # 1. Check our config file first
    config = load_config()
    hf_config = config.get("hf", {})
    if api_key := hf_config.get("api_key"):
        return api_key

    # 2. Check environment variable
    if env_token := os.environ.get("HF_TOKEN"):
        return env_token

    # 3. Check huggingface_hub token file
    try:
        from huggingface_hub import get_token

        return get_token()
    except ImportError:
        pass

    return None


def has_hf_token() -> bool:
    """Check if HF token is available from any source."""
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


def update_api_key_in_config(api_key: str) -> None:
    """Update the hf.api_key in the config file.

    This stores the HuggingFace token in the config file so the LLM provider
    can access it via the standard ProviderKeyManager mechanism.

    Args:
        api_key: The HuggingFace API token
    """
    config_path = ensure_config_exists()
    config = load_config()

    # Ensure hf section exists
    if "hf" not in config:
        config["hf"] = {}

    config["hf"]["api_key"] = api_key

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def get_api_key_from_config() -> str | None:
    """Get the hf.api_key from the config file.

    Returns:
        The API key if set, None otherwise
    """
    config = load_config()
    hf_config = config.get("hf", {})
    return hf_config.get("api_key")


def get_default_model() -> str:
    """Get the default model from config, or return the default.

    Returns:
        The default model name
    """
    config = load_config()
    return config.get("default_model", DEFAULT_MODEL)
