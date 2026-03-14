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
_MODEL_REFERENCE_PATTERN = re.compile(
    r"^\$(?P<namespace>[A-Za-z_][A-Za-z0-9_-]*)\.(?P<key>[A-Za-z_][A-Za-z0-9_-]*)$"
)
logger = get_logger(__name__)


def _is_system_default_alias(value: str | None) -> bool:
    """Return True when a value is the special ``$system.default`` reference token."""
    return value is not None and value.strip() == "$system.default"


def parse_model_reference_token(token: str) -> tuple[str, str]:
    """Parse and validate a model reference token.

    Args:
        token: Reference token in ``$<namespace>.<key>`` format.

    Returns:
        ``(namespace, key)`` tuple.

    Raises:
        ModelConfigError: If token format is invalid.
    """
    normalized = token.strip()
    match = _MODEL_REFERENCE_PATTERN.fullmatch(normalized)
    if match is None:
        raise ModelConfigError(
            f"Invalid model reference '{normalized}'",
            "Model references must be exact tokens in the format '$<namespace>.<key>' "
            "(for example '$system.fast').",
        )
    return match.group("namespace"), match.group("key")


def resolve_model_reference(
    model: str,
    references: Mapping[str, Mapping[str, str]] | None,
) -> str:
    """Resolve a model reference token like ``$system.fast`` to its model string.

    Phase 1 intentionally supports only exact reference tokens. Values may recursively
    point to other reference tokens and are expanded with cycle protection.
    """
    model_spec = model.strip()
    if not model_spec.startswith("$"):
        return model_spec

    return _resolve_reference_recursive(model_spec, references, stack=[])


def _resolve_reference_recursive(
    token: str,
    references: Mapping[str, Mapping[str, str]] | None,
    *,
    stack: list[str],
) -> str:
    namespace, key = parse_model_reference_token(token)

    if references is None or len(references) == 0:
        raise ModelConfigError(
            f"Model reference '{token}' could not be resolved",
            "No model_references are configured. Add a model_references section in fastagent.config.yaml.",
        )

    if token in stack:
        cycle = " -> ".join([*stack, token])
        raise ModelConfigError(
            "Model reference cycle detected",
            f"Detected reference cycle: {cycle}",
        )

    namespace_map = references.get(namespace)

    if namespace_map is None:
        available_namespaces = ", ".join(sorted(references.keys()))
        details = f"Unknown namespace '{namespace}'."
        if available_namespaces:
            details += f" Available namespaces: {available_namespaces}."
        raise ModelConfigError(f"Model reference '{token}' could not be resolved", details)

    raw_value = namespace_map.get(key)
    if raw_value is None:
        available_keys = ", ".join(sorted(namespace_map.keys()))
        details = f"Unknown key '{key}' in namespace '{namespace}'."
        if available_keys:
            details += f" Available keys: {available_keys}."
        raise ModelConfigError(f"Model reference '{token}' could not be resolved", details)

    value = raw_value.strip()
    if not value:
        raise ModelConfigError(
            f"Model reference '{token}' could not be resolved",
            f"Reference '{namespace}.{key}' maps to an empty value.",
        )

    if not value.startswith("$"):
        return value

    return _resolve_reference_recursive(value, references, stack=[*stack, token])


def get_context_model_references(
    context: Any,
) -> Mapping[str, Mapping[str, str]] | None:
    """Return configured model references from context, if available."""
    if not context:
        return None
    config = getattr(context, "config", None)
    if not config:
        return None
    model_references = getattr(config, "model_references", None)
    return model_references if isinstance(model_references, Mapping) else None


def resolve_model_spec(
    context: Any,
    model: str | None = None,
    default_model: str | None = None,
    cli_model: str | None = None,
    *,
    env_var: str = "FAST_AGENT_MODEL",
    hardcoded_default: str | None = None,
    fallback_to_hardcoded: bool = True,
    model_references: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[str | None, str | None]:
    """
    Resolve the model specification and report the source used.

    Precedence (lowest to highest):
        1. Hardcoded default (if enabled)
        2. Environment variable
        3. Config file default_model
        4. CLI --model argument
        5. Explicit model parameter

    Special case: explicit ``$system.default`` is treated as a "use current
    default" placeholder, so it is evaluated *after* CLI ``--model`` but before
    config/env/hardcoded fallbacks.
    """
    candidates: list[tuple[str, str]] = []

    def _add_candidate(value: str | None, source_label: str) -> None:
        if value is None:
            return
        stripped = value.strip()
        if not stripped:
            return
        candidates.append((stripped, source_label))

    explicit_is_system_default = _is_system_default_alias(model)

    if not explicit_is_system_default:
        _add_candidate(model, "explicit model")

    _add_candidate(cli_model, "CLI --model")

    # ``$system.default`` is an explicit placeholder for "use the current default".
    # Keep it above config/env/hardcoded defaults, but below CLI overrides.
    if explicit_is_system_default:
        _add_candidate(model, "explicit model")

    config_default = default_model
    if config_default is None and context and getattr(context, "config", None):
        config_default = context.config.default_model
    _add_candidate(config_default, "config file")

    env_model = os.getenv(env_var)
    _add_candidate(env_model, f"environment variable {env_var}")

    if fallback_to_hardcoded:
        _add_candidate(hardcoded_default, "hardcoded default")

    references = (
        model_references
        if model_references is not None
        else get_context_model_references(context)
    )

    for index, (candidate, source) in enumerate(candidates):
        try:
            return resolve_model_reference(candidate, references), source
        except ModelConfigError as exc:
            if not candidate.startswith("$"):
                raise

            fallback_source = next((label for _, label in candidates[index + 1 :]), None)
            logger.warning(
                "Failed to resolve model reference; trying lower-precedence default",
                model_reference=candidate,
                source=source,
                fallback_source=fallback_source,
                error=exc.message,
                details=exc.details,
            )

    return None, None
