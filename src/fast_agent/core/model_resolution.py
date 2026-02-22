"""
Shared model resolution helpers to avoid circular imports.
"""

import os
import re
from collections.abc import Mapping
from typing import Any

from fast_agent.core.exceptions import ModelConfigError
from fast_agent.core.logging.logger import get_logger

HARDCODED_DEFAULT_MODEL = "gpt-5-mini?reasoning=low"
_MODEL_ALIAS_PATTERN = re.compile(
    r"^\$(?P<namespace>[A-Za-z_][A-Za-z0-9_-]*)\.(?P<key>[A-Za-z_][A-Za-z0-9_-]*)$"
)
logger = get_logger(__name__)


def parse_model_alias_token(token: str) -> tuple[str, str]:
    """Parse and validate a model alias token.

    Args:
        token: Alias token in ``$<namespace>.<key>`` format.

    Returns:
        ``(namespace, key)`` tuple.

    Raises:
        ModelConfigError: If token format is invalid.
    """
    normalized = token.strip()
    match = _MODEL_ALIAS_PATTERN.fullmatch(normalized)
    if match is None:
        raise ModelConfigError(
            f"Invalid model alias '{normalized}'",
            "Model aliases must be exact tokens in the format '$<namespace>.<key>' "
            "(for example '$system.fast').",
        )
    return match.group("namespace"), match.group("key")


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
    namespace, key = parse_model_alias_token(token)

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
    candidates: list[tuple[str, str]] = []

    def _add_candidate(value: str | None, source_label: str) -> None:
        if value is None:
            return
        stripped = value.strip()
        if not stripped:
            return
        candidates.append((stripped, source_label))

    _add_candidate(model, "explicit model")
    _add_candidate(cli_model, "CLI --model")

    config_default = default_model
    if config_default is None and context and getattr(context, "config", None):
        config_default = context.config.default_model
    _add_candidate(config_default, "config file")

    env_model = os.getenv(env_var)
    _add_candidate(env_model, f"environment variable {env_var}")

    if fallback_to_hardcoded:
        _add_candidate(hardcoded_default, "hardcoded default")

    aliases = model_aliases if model_aliases is not None else get_context_model_aliases(context)

    for index, (candidate, source) in enumerate(candidates):
        try:
            return resolve_model_alias(candidate, aliases), source
        except ModelConfigError as exc:
            if not candidate.startswith("$"):
                raise

            fallback_source = next((label for _, label in candidates[index + 1 :]), None)
            logger.warning(
                "Failed to resolve model alias; trying lower-precedence default",
                model_alias=candidate,
                source=source,
                fallback_source=fallback_source,
                error=exc.message,
                details=exc.details,
            )

    return None, None
