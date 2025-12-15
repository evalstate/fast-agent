"""Configuration file handling for hf-inference-acp."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

CONFIG_DIR = Path.home() / ".config" / "hf-inference"
CONFIG_FILE = CONFIG_DIR / "hf.config.yaml"
SYSTEM_PROMPT_FILE = CONFIG_DIR / "hf.system_prompt.md"

DEFAULT_MODEL = "kimi"

# Use round-trip mode to preserve comments and formatting
_yaml = YAML()
_yaml.preserve_quotes = True


def get_hf_token() -> str | None:
    """Get HF token from all available sources.

    Checks in priority order:
    1. Our config file (hf.api_key)
    2. HF_TOKEN environment variable
    3. huggingface_hub token file (~/.cache/huggingface/token)
    """
    token, _ = discover_hf_token()
    return token


def discover_hf_token(*, ignore_env: bool = False) -> tuple[str | None, str | None]:
    """
    Discover the HF token and report where it came from.

    Returns:
        (token, source) where source is one of: "config", "env", "huggingface_hub", or None
    """
    # 1. Check our config file first
    config = load_config()
    hf_config = config.get("hf") or {}
    if api_key := hf_config.get("api_key"):
        return api_key, "config"

    # 2. Check environment variable
    if not ignore_env:
        if env_token := os.environ.get("HF_TOKEN"):
            return env_token, "env"

    # 3. Check huggingface_hub token file
    try:
        from huggingface_hub import get_token

        token = get_token()
        return token, "huggingface_hub" if token else (None, None)
    except ImportError:
        pass

    return None, None


def get_hf_token_source(*, ignore_env: bool = False) -> str | None:
    """Return the discovered HF token source without returning the token itself."""
    _, source = discover_hf_token(ignore_env=ignore_env)
    return source


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
        resource_path = files("hf_inference_acp").joinpath("resources").joinpath("hf.config.yaml")
        if resource_path.is_file():
            template_content = resource_path.read_text()
            CONFIG_FILE.write_text(template_content)

    return CONFIG_FILE


def ensure_system_prompt_exists() -> Path:
    """Ensure system prompt file exists, creating from template if needed.

    The system prompt is copied to the config directory so users can customize it.

    Returns:
        Path to the system prompt file
    """
    from importlib.resources import files

    # Ensure config directory exists
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Create system prompt from template if it doesn't exist
    if not SYSTEM_PROMPT_FILE.exists():
        resource_path = (
            files("hf_inference_acp").joinpath("resources").joinpath("hf.system_prompt.md")
        )
        if resource_path.is_file():
            template_content = resource_path.read_text()
            SYSTEM_PROMPT_FILE.write_text(template_content)

    return SYSTEM_PROMPT_FILE


def load_system_prompt() -> str:
    """Load the system prompt from the config directory.

    Returns:
        The system prompt content, or a fallback if not available
    """
    prompt_path = ensure_system_prompt_exists()

    if prompt_path.exists():
        return prompt_path.read_text()

    # Fallback
    return """You are a helpful AI assistant powered by Hugging Face Inference API.

{{file_silent:AGENTS.md}}
{{file_silent:huggingface.md}}
"""


def load_config() -> dict[str, Any]:
    """Load configuration from the config file.

    Returns:
        Configuration dictionary (as ruamel.yaml CommentedMap to preserve comments)
    """
    config_path = ensure_config_exists()

    if config_path.exists():
        with open(config_path) as f:
            return _yaml.load(f) or {}

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
        _yaml.dump(config, f)


def get_api_key_from_config() -> str | None:
    """Get the hf.api_key from the config file.

    Returns:
        The API key if set, None otherwise
    """
    config = load_config()
    hf_config = config.get("hf") or {}
    return hf_config.get("api_key")


def get_default_model() -> str:
    """Get the default model from config, or return the default.

    Returns:
        The default model name
    """
    config = load_config()
    return config.get("default_model", DEFAULT_MODEL)


def update_mcp_server_load_on_start(server_name: str, load_on_start: bool) -> None:
    """Update the load_on_start setting for an MCP server.

    Args:
        server_name: The name of the MCP server
        load_on_start: Whether to load the server on start
    """
    config_path = ensure_config_exists()
    config = load_config()

    # Ensure nested structure exists
    if "mcp" not in config or config["mcp"] is None:
        config["mcp"] = {}
    if "servers" not in config["mcp"] or config["mcp"]["servers"] is None:
        config["mcp"]["servers"] = {}
    if server_name not in config["mcp"]["servers"] or config["mcp"]["servers"][server_name] is None:
        config["mcp"]["servers"][server_name] = {}

    config["mcp"]["servers"][server_name]["load_on_start"] = load_on_start

    with open(config_path, "w") as f:
        _yaml.dump(config, f)
