"""
Shared model resolution helpers to avoid circular imports.
"""

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from fast_agent.core.exceptions import ModelConfigError
from fast_agent.core.logging.logger import get_logger
from fast_agent.utils.text import strip_str_to_none, strip_to_none

if TYPE_CHECKING:
    from fast_agent.context import Context

HARDCODED_DEFAULT_MODEL = "gpt-5.4-mini?reasoning=low"
_MODEL_REFERENCE_PATTERN = re.compile(
    r"^\$(?P<namespace>[A-Za-z_][A-Za-z0-9_-]*)\.(?P<key>[A-Za-z_][A-Za-z0-9_-]*)$"
)
logger = get_logger(__name__)


@runtime_checkable
class _ModelReferencesConfig(Protocol):
    model_references: object


@runtime_checkable
class _CliModelOverrideConfig(Protocol):
    cli_model_override: object


@dataclass(frozen=True, slots=True)
class ResolvedModelSpec:
    model: str | None = None
    source: str | None = None


type ModelCandidate = tuple[str, str]


def _is_system_default_alias(value: str | None) -> bool:
    """Return True when a value is the special ``$system.default`` reference token."""
    return strip_to_none(value) == "$system.default"


def parse_model_reference_token(token: str) -> tuple[str, str]:
    """Parse and validate a model reference token.

    Args:
        token: Reference token in ``$<namespace>.<key>`` format.

    Returns:
        ``(namespace, key)`` tuple.

    Raises:
        ModelConfigError: If token format is invalid.
    """
    normalized = strip_to_none(token) or ""
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
    model_spec = strip_to_none(model) or ""
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

    if not references:
        raise ModelConfigError(
            f"Model reference '{token}' could not be resolved",
            "No model_references are configured. Add a model_references section in fast-agent.yaml.",
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

    value = strip_to_none(raw_value)
    if value is None:
        raise ModelConfigError(
            f"Model reference '{token}' could not be resolved",
            f"Reference '{namespace}.{key}' maps to an empty value.",
        )

    if not value.startswith("$"):
        return value

    return _resolve_reference_recursive(value, references, stack=[*stack, token])


def get_context_model_references(
    context: "Context | None",
) -> Mapping[str, Mapping[str, str]] | None:
    """Return configured model references from context, if available."""
    if context is None or context.config is None:
        return None
    if not isinstance(context.config, _ModelReferencesConfig):
        return None
    model_references = context.config.model_references
    return model_references if isinstance(model_references, Mapping) else None


def get_context_cli_model_override(context: "Context | None") -> str | None:
    """Return the current run's CLI/model-picker override from context, if any."""
    if context is None or context.config is None:
        return None
    if not isinstance(context.config, _CliModelOverrideConfig):
        return None
    cli_model = context.config.cli_model_override
    return strip_str_to_none(cli_model)


def _model_resolution_candidates(
    *,
    context: "Context | None",
    model: str | None,
    default_model: str | None,
    cli_model: str | None,
    env_var: str,
    hardcoded_default: str | None,
    fallback_to_hardcoded: bool,
) -> list[ModelCandidate]:
    candidates: list[ModelCandidate] = []

    def add_candidate(value: str | None, source_label: str) -> None:
        stripped = strip_to_none(value)
        if stripped is not None:
            candidates.append((stripped, source_label))

    explicit_is_system_default = _is_system_default_alias(model)

    if not explicit_is_system_default:
        add_candidate(model, "explicit model")

    add_candidate(cli_model, "CLI --model")

    # ``$system.default`` is an explicit placeholder for "use the current default".
    # Keep it above config/env/hardcoded defaults, but below CLI overrides.
    if explicit_is_system_default:
        add_candidate(model, "explicit model")

    config_default = _config_default_model(context, default_model)
    add_candidate(config_default, "config file")

    add_candidate(os.getenv(env_var), f"environment variable {env_var}")

    if fallback_to_hardcoded:
        add_candidate(hardcoded_default, "hardcoded default")

    return candidates


def _config_default_model(context: "Context | None", default_model: str | None) -> str | None:
    if default_model is not None:
        return default_model
    if context is not None and context.config is not None:
        return context.config.default_model
    return None


def _candidate_fallback_source(candidates: list[ModelCandidate], index: int) -> str | None:
    return next((label for _, label in candidates[index + 1 :]), None)


def _log_model_reference_fallback(
    *,
    candidate: str,
    source: str,
    fallback_source: str | None,
    error: ModelConfigError,
) -> None:
    logger.warning(
        "Failed to resolve model reference; trying lower-precedence default",
        model_reference=candidate,
        source=source,
        fallback_source=fallback_source,
        error=error.message,
        details=error.details,
    )


def resolve_model_spec(
    context: "Context | None",
    model: str | None = None,
    default_model: str | None = None,
    cli_model: str | None = None,
    *,
    env_var: str = "FAST_AGENT_MODEL",
    hardcoded_default: str | None = None,
    fallback_to_hardcoded: bool = True,
    model_references: Mapping[str, Mapping[str, str]] | None = None,
) -> ResolvedModelSpec:
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
    candidates = _model_resolution_candidates(
        context=context,
        model=model,
        default_model=default_model,
        cli_model=cli_model,
        env_var=env_var,
        hardcoded_default=hardcoded_default,
        fallback_to_hardcoded=fallback_to_hardcoded,
    )
    references = (
        model_references if model_references is not None else get_context_model_references(context)
    )

    for index, (candidate, source) in enumerate(candidates):
        try:
            return ResolvedModelSpec(resolve_model_reference(candidate, references), source)
        except ModelConfigError as exc:
            if not candidate.startswith("$"):
                raise

            _log_model_reference_fallback(
                candidate=candidate,
                source=source,
                fallback_source=_candidate_fallback_source(candidates, index),
                error=exc,
            )

    return ResolvedModelSpec()
