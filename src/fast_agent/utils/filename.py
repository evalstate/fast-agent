"""Filename-safe string helpers."""

from __future__ import annotations

import re

_FILENAME_COMPONENT_SAFE_CHARS = frozenset(("-", "_", "."))
_WINDOWS_RESERVED_FILENAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{index}" for index in range(1, 10)),
        *(f"LPT{index}" for index in range(1, 10)),
    }
)
_PATH_OR_SPACE_SEPARATOR_PATTERN = re.compile(r"[\\/\s]+")
_REPEATED_UNDERSCORE_PATTERN = re.compile(r"_+")


def _is_safe_filename_component_char(char: str) -> bool:
    return char.isalnum() or char in _FILENAME_COMPONENT_SAFE_CHARS


def _replace_unsafe_filename_component_chars(value: str) -> str:
    return "".join(
        char if _is_safe_filename_component_char(char) else "_" for char in value
    )


def _clean_sanitized_filename(value: str) -> str:
    return _REPEATED_UNDERSCORE_PATTERN.sub("_", value).strip("._-")


def _avoid_reserved_filename(value: str, *, fallback: str) -> str:
    stem = value.split(".", maxsplit=1)[0].upper()
    if stem in _WINDOWS_RESERVED_FILENAMES:
        return fallback
    return value


def _finalize_sanitized_filename(value: str, *, fallback: str) -> str:
    cleaned = _clean_sanitized_filename(value)
    if not cleaned:
        return fallback
    return _avoid_reserved_filename(cleaned, fallback=fallback)


def sanitize_filename_component(value: str, *, fallback: str) -> str:
    sanitized = _replace_unsafe_filename_component_chars(value)
    return _finalize_sanitized_filename(sanitized, fallback=fallback)


def sanitize_filename_suffix(label: str, *, fallback: str = "agent") -> str:
    normalized = _PATH_OR_SPACE_SEPARATOR_PATTERN.sub("_", label.strip())
    sanitized = _replace_unsafe_filename_component_chars(normalized)
    return _finalize_sanitized_filename(sanitized, fallback=fallback)
