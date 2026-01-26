"""Text verbosity settings shared across providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, cast

TextVerbosityLevel: TypeAlias = Literal["low", "medium", "high"]

TEXT_VERBOSITY_LEVELS: Final[tuple[TextVerbosityLevel, ...]] = (
    "low",
    "medium",
    "high",
)

VERBOSITY_ALIASES: Final[dict[str, TextVerbosityLevel]] = {
    "med": "medium",
}


@dataclass(frozen=True, slots=True)
class TextVerbositySpec:
    """Capability info describing how a model accepts text verbosity."""

    allowed: tuple[TextVerbosityLevel, ...] = TEXT_VERBOSITY_LEVELS
    default: TextVerbosityLevel = "medium"


def parse_text_verbosity(value: str | None) -> TextVerbosityLevel | None:
    """Parse a text verbosity value from raw input."""
    if value is None:
        return None
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    normalized = VERBOSITY_ALIASES.get(cleaned, cleaned)
    if normalized in TEXT_VERBOSITY_LEVELS:
        return cast("TextVerbosityLevel", normalized)
    return None


def validate_text_verbosity(
    value: TextVerbosityLevel,
    spec: TextVerbositySpec | None,
) -> TextVerbosityLevel:
    """Validate a text verbosity value against a model spec."""
    if spec is None:
        return value
    if value in spec.allowed:
        return value
    allowed = ", ".join(spec.allowed) or "any"
    raise ValueError(f"Verbosity '{value}' not allowed (allowed: {allowed}).")


def format_text_verbosity(value: TextVerbosityLevel | None) -> str:
    if value is None:
        return "unset"
    return value


def available_text_verbosity_values(spec: TextVerbositySpec | None) -> list[str]:
    if spec is None:
        return []
    return list(spec.allowed)
