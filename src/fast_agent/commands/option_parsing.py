"""Shared helpers for command option parsing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, Self, TypeVar

from fast_agent.utils.text import strip_to_none

if TYPE_CHECKING:
    from collections.abc import Sequence

OptionNameT = TypeVar("OptionNameT")


@dataclass(frozen=True, slots=True)
class ValueOption(Generic[OptionNameT]):
    name: OptionNameT
    option_names: tuple[str, ...]
    error_name: str | None = None

    @property
    def display_name(self) -> str:
        return self.error_name or self.option_names[0]


@dataclass(frozen=True, slots=True)
class ParsedValueOption(Generic[OptionNameT]):
    name: OptionNameT | None = None
    matched: bool = False
    value: str | None = None
    display_name: str | None = None
    error: str | None = None
    next_index: int = 0

    def require_name(self) -> OptionNameT:
        if self.name is None:
            raise ValueError("option name was not parsed")
        return self.name

    def require_value(self) -> str:
        if self.value is None:
            raise ValueError("option value was not parsed")
        return self.value


@dataclass(frozen=True, slots=True)
class ParsedOptionTokenValue:
    matched: bool = False
    value: str | None = None
    error: str | None = None
    next_index: int = 0

    @classmethod
    def unmatched(cls, *, next_index: int) -> Self:
        return cls(next_index=next_index)

    @classmethod
    def missing_value(cls, option_name: str, *, next_index: int) -> Self:
        return cls(
            matched=True,
            error=_missing_value_error(option_name),
            next_index=next_index,
        )

    @classmethod
    def parsed(cls, value: str, *, next_index: int) -> Self:
        return cls(
            matched=True,
            value=value,
            next_index=next_index,
        )

    def require_value(self) -> str:
        if self.value is None:
            raise ValueError("option value was not parsed")
        return self.value


def is_long_option_token(token: str) -> bool:
    """Return whether a token should be treated as a long option/flag."""
    return len(token) > 2 and token.startswith("--") and token[2] != "="


def matches_option_token(token: str, option_names: tuple[str, ...]) -> bool:
    for option_name in option_names:
        if token == option_name:
            return True
        if option_name.startswith("--") and token.startswith(f"{option_name}="):
            return True
    return False


def _missing_value_error(option_name: str) -> str:
    return f"Missing value for {option_name}"


def _is_missing_option_value(
    value: str,
    *,
    allow_flag_like_value: bool,
) -> bool:
    if strip_to_none(value) is None:
        return True
    return value.startswith("-") and not allow_flag_like_value


def _accepted_option_value(value: str) -> str:
    return strip_to_none(value) or value


def _read_split_option_value(
    tokens: Sequence[str],
    index: int,
    *,
    option_name: str,
    allow_flag_like_value: bool,
) -> ParsedOptionTokenValue:
    value_index = index + 1
    if value_index >= len(tokens):
        return ParsedOptionTokenValue.missing_value(option_name, next_index=index)

    candidate = tokens[value_index]
    if _is_missing_option_value(
        candidate,
        allow_flag_like_value=allow_flag_like_value,
    ):
        return ParsedOptionTokenValue.missing_value(option_name, next_index=index)

    return ParsedOptionTokenValue.parsed(
        _accepted_option_value(candidate),
        next_index=index + 2,
    )


def _read_equals_option_value(
    token: str,
    index: int,
    *,
    option_name: str,
) -> ParsedOptionTokenValue:
    value = token.split("=", maxsplit=1)[1]
    if _is_missing_option_value(value, allow_flag_like_value=True):
        return ParsedOptionTokenValue.missing_value(option_name, next_index=index)
    return ParsedOptionTokenValue.parsed(
        _accepted_option_value(value),
        next_index=index + 1,
    )


def read_option_token_value(
    tokens: Sequence[str],
    index: int,
    option_names: tuple[str, ...],
    *,
    error_name: str | None = None,
    allow_flag_like_value: bool = False,
) -> ParsedOptionTokenValue:
    if index < 0 or index >= len(tokens):
        return ParsedOptionTokenValue.unmatched(next_index=index)

    token = tokens[index]
    for option_name in option_names:
        option_error_name = error_name or option_name
        if token == option_name:
            return _read_split_option_value(
                tokens,
                index,
                option_name=option_error_name,
                allow_flag_like_value=allow_flag_like_value,
            )

        if matches_option_token(token, (option_name,)):
            return _read_equals_option_value(
                token,
                index,
                option_name=option_error_name,
            )

    return ParsedOptionTokenValue.unmatched(next_index=index)


def read_value_option(
    tokens: Sequence[str],
    index: int,
    options: Sequence[ValueOption[OptionNameT]],
    *,
    allow_flag_like_value: bool = False,
) -> ParsedValueOption[OptionNameT]:
    for option in options:
        parsed = read_option_token_value(
            tokens,
            index,
            option.option_names,
            error_name=option.error_name,
            allow_flag_like_value=allow_flag_like_value,
        )
        if not parsed.matched:
            continue
        return ParsedValueOption(
            name=option.name,
            matched=True,
            value=parsed.value,
            display_name=option.display_name,
            error=parsed.error,
            next_index=parsed.next_index,
        )
    return ParsedValueOption(next_index=index)
