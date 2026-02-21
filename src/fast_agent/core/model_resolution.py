"""
Shared model resolution helpers to avoid circular imports.
"""

import os
import re
from collections.abc import Mapping
from typing import Any

from fast_agent.core.exceptions import ModelConfigError

HARDCODED_DEFAULT_MODEL = "gpt-5-mini?reasoning=low"
_MODEL_ALIAS_PATTERN = re.compile(
    r"^\$(?P<namespace>[A-Za-z_][A-Za-z0-9_-]*)\.(?P<key>[A-Za-z_][A-Za-z0-9_-]*)$"
)


def resolve_model_alias(
    model: str,
    aliases: Mapping[str, Mapping[str, str]] | None,
) -> str:
    """Resolve a model alias token like ``$system.fast`` to its model string.

    Phase 1 intentionally supports only exact alias tokens. Values may recursively
    point to other alias tokens and are expanded with cycle protection.
    """
    model_spec = model.strip()
    if not model_spec.startswith("$"):
        return model_spec

    return _resolve_alias_recursive(model_spec, aliases, stack=[])


def _resolve_alias_recursive(
    token: str,
    aliases: Mapping[str, Mapping[str, str]] | None,
    *,
    stack: list[str],
) -> str:
    match = _MODEL_ALIAS_PATTERN.fullmatch(token)
    if match is None:
        raise ModelConfigError(
            f"Invalid model alias '{token}'",
            "Model aliases must be exact tokens in the format '$<namespace>.<key>' "
            "(for example '$system.fast').",
        )

    if aliases is None or len(aliases) == 0:
        raise ModelConfigError(
            f"Model alias '{token}' could not be resolved",
            "No model_aliases are configured. Add a model_aliases section in fastagent.config.yaml.",
        )

    if token in stack:
        cycle = " -> ".join([*stack, token])
        raise ModelConfigError(
            "Model alias cycle detected",
            f"Detected alias cycle: {cycle}",
        )

    namespace = match.group("namespace")
    key = match.group("key")
    namespace_map = aliases.get(namespace)

    if namespace_map is None:
        available_namespaces = ", ".join(sorted(aliases.keys()))
        details = f"Unknown namespace '{namespace}'."
        if available_namespaces:
            details += f" Available namespaces: {available_namespaces}."
        raise ModelConfigError(f"Model alias '{token}' could not be resolved", details)

    raw_value = namespace_map.get(key)
    if raw_value is None:
        available_keys = ", ".join(sorted(namespace_map.keys()))
        details = f"Unknown key '{key}' in namespace '{namespace}'."
        if available_keys:
            details += f" Available keys: {available_keys}."
        raise ModelConfigError(f"Model alias '{token}' could not be resolved", details)

    value = raw_value.strip()
    if not value:
        raise ModelConfigError(
            f"Model alias '{token}' could not be resolved",
            f"Alias '{namespace}.{key}' maps to an empty value.",
        )

    if not value.startswith("$"):
        return value

    return _resolve_alias_recursive(value, aliases, stack=[*stack, token])


def get_context_model_aliases(
    context: Any,
) -> Mapping[str, Mapping[str, str]] | None:
    """Return configured model aliases from context, if available."""
    if not context:
        return None
    config = getattr(context, "config", None)
    if not config:
        return None
    model_aliases = getattr(config, "model_aliases", None)
    return model_aliases if isinstance(model_aliases, Mapping) else None


def resolve_model_spec(
    context: Any,
    model: str | None = None,
    default_model: str | None = None,
    cli_model: str | None = None,
    *,
    env_var: str = "FAST_AGENT_MODEL",
    hardcoded_default: str | None = None,
    fallback_to_hardcoded: bool = True,
    model_aliases: Mapping[str, Mapping[str, str]] | None = None,
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

    aliases = model_aliases if model_aliases is not None else get_context_model_aliases(context)
    if model_spec:
        model_spec = resolve_model_alias(model_spec, aliases)

    return model_spec, source
