"""
Shared model resolution helpers to avoid circular imports.
"""

import os
from typing import Any

HARDCODED_DEFAULT_MODEL = "gpt-5-mini.low"


def resolve_model_spec(
    context: Any,
    model: str | None = None,
    default_model: str | None = None,
    cli_model: str | None = None,
    *,
    env_var: str = "FAST_AGENT_MODEL",
    hardcoded_default: str | None = None,
    fallback_to_hardcoded: bool = True,
) -> tuple[str | None, str | None]:
    """
    Resolve the model specification and report the source used.

    Precedence (lowest to highest):
        1. Hardcoded default (if enabled)
        2. Environment variable
        3. Config file default_model
        4. CLI --model argument
        5. Explicit model parameter
    """
    model_spec: str | None = hardcoded_default if fallback_to_hardcoded else None
    source: str | None = "hardcoded default" if fallback_to_hardcoded else None

    env_model = os.getenv(env_var)
    if env_model:
        model_spec = env_model
        source = f"environment variable {env_var}"

    config_default = default_model
    if config_default is None and context and getattr(context, "config", None):
        config_default = context.config.default_model
    if config_default:
        model_spec = config_default
        source = "config file"

    if cli_model:
        model_spec = cli_model
        source = "CLI --model"

    if model:
        model_spec = model
        source = "explicit model"

    return model_spec, source
