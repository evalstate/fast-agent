"""Conservative formatting for JSON object and array stream prefixes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

_WHITESPACE = frozenset(" \t\r\n")
_SIMPLE_ESCAPES = frozenset('"\\/bfnrt')
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")

Expectation = Literal["key_or_end", "colon", "value", "value_or_end", "comma_or_end"]
LexicalMode = Literal["normal", "string", "scalar"]
NumberState = Literal[
    "sign",
    "zero",
    "integer",
    "decimal_point",
    "fraction",
    "exponent_mark",
    "exponent_sign",
    "exponent",
]


@dataclass
class _Frame:
    opener: str
    expectation: Expectation
    after_comma: bool = False


class JsonPrefixFormatter:
    """Incrementally pretty-format one JSON object or array prefix."""

    def __init__(self, *, indent: int = 2) -> None:
        if indent < 0:
            raise ValueError("indent must be non-negative")
        self.indent = indent
        self._frames: list[_Frame] = []
        self._result: list[str] = []
        self._root_started = False
        self._root_complete = False
        self._invalid = False
        self._mode: LexicalMode = "normal"
        self._string_escape = False
        self._unicode_digits_remaining = 0
        self._literal = ""
        self._literal_target: str | None = None
        self._number_state: NumberState | None = None
        self._depth = 0
        self._pending_line = False
        self._previous_kind: Literal["open", "close", "colon", "comma", "value"] | None = None
        self._formatted_cache: str | None = None
        self._formatted_dirty = False

    @property
    def formatted(self) -> str | None:
        if self._invalid or not self._root_started:
            return None
        if self._formatted_dirty or self._formatted_cache is None:
            self._formatted_cache = "".join(self._result)
            self._formatted_dirty = False
        return self._formatted_cache

    def append(self, chunk: str) -> str | None:
        """Append source text and return the formatted prefix, or ``None`` if invalid."""
        if not chunk or self._invalid:
            return self.formatted

        index = 0
        while index < len(chunk):
            char = chunk[index]
            if self._mode == "string":
                if not self._append_string_character(char):
                    self._invalid = True
                    return None
                index += 1
                continue
            if self._mode == "scalar":
                if char not in _WHITESPACE and char not in "{}[]:,":
                    if not self._append_scalar_character(char):
                        self._invalid = True
                        return None
                    self._result.append(char)
                    self._formatted_dirty = True
                    index += 1
                    continue
                if not self._scalar_complete():
                    self._invalid = True
                    return None
                self._mode = "normal"
                self._literal = ""
                self._literal_target = None
                self._number_state = None
                continue
            if not self._append_normal_character(char):
                self._invalid = True
                return None
            index += 1
        return self.formatted

    def _append_normal_character(self, char: str) -> bool:
        if char in _WHITESPACE:
            return True
        if self._root_complete:
            return False
        if not self._root_started and char not in "{[":
            return False
        if char in "{[":
            return self._open_container(char)
        if char in "}]":
            return self._close_container(char)
        if char == ":":
            return self._append_colon()
        if char == ",":
            return self._append_comma()
        if char == '"':
            return self._start_string()
        return self._start_scalar(char)

    def _open_container(self, opener: str) -> bool:
        if not self._consume_value():
            return False
        self._emit_value_start()
        self._result.append(opener)
        self._formatted_dirty = True
        self._depth += 1
        self._pending_line = True
        self._previous_kind = "open"
        self._frames.append(
            _Frame(
                opener=opener,
                expectation="key_or_end" if opener == "{" else "value_or_end",
            )
        )
        return True

    def _close_container(self, closer: str) -> bool:
        if not self._frames:
            return False
        frame = self._frames[-1]
        if (frame.opener == "{" and closer != "}") or (frame.opener == "[" and closer != "]"):
            return False
        if frame.after_comma or frame.expectation not in {
            "key_or_end",
            "value_or_end",
            "comma_or_end",
        }:
            return False

        self._frames.pop()
        self._depth -= 1
        if self._previous_kind != "open":
            self._result.extend(("\n", " " * (self._depth * self.indent)))
        self._result.append(closer)
        self._formatted_dirty = True
        self._pending_line = False
        self._previous_kind = "close"
        if not self._frames:
            self._root_complete = True
        return True

    def _append_colon(self) -> bool:
        if not self._frames:
            return False
        frame = self._frames[-1]
        if frame.opener != "{" or frame.expectation != "colon":
            return False
        frame.expectation = "value"
        self._result.append(": ")
        self._formatted_dirty = True
        self._pending_line = False
        self._previous_kind = "colon"
        return True

    def _append_comma(self) -> bool:
        if not self._frames or self._frames[-1].expectation != "comma_or_end":
            return False
        frame = self._frames[-1]
        frame.expectation = "key_or_end" if frame.opener == "{" else "value_or_end"
        frame.after_comma = True
        self._result.append(",")
        self._formatted_dirty = True
        self._pending_line = True
        self._previous_kind = "comma"
        return True

    def _start_string(self) -> bool:
        if self._consume_key():
            pass
        elif not self._consume_value():
            return False
        self._emit_value_start()
        self._result.append('"')
        self._formatted_dirty = True
        self._mode = "string"
        self._string_escape = False
        self._unicode_digits_remaining = 0
        self._previous_kind = "value"
        return True

    def _append_string_character(self, char: str) -> bool:
        if ord(char) < 0x20:
            return False
        if self._unicode_digits_remaining:
            if char not in _HEX_DIGITS:
                return False
            self._unicode_digits_remaining -= 1
            self._result.append(char)
            self._formatted_dirty = True
            return True
        if self._string_escape:
            if char in _SIMPLE_ESCAPES:
                self._string_escape = False
            elif char == "u":
                self._string_escape = False
                self._unicode_digits_remaining = 4
            else:
                return False
            self._result.append(char)
            self._formatted_dirty = True
            return True
        if char == "\\":
            self._string_escape = True
            self._result.append(char)
            self._formatted_dirty = True
            return True
        self._result.append(char)
        self._formatted_dirty = True
        if char == '"':
            self._mode = "normal"
        return True

    def _start_scalar(self, char: str) -> bool:
        if not self._consume_value() or not self._initialize_scalar(char):
            return False
        self._emit_value_start()
        self._result.append(char)
        self._formatted_dirty = True
        self._mode = "scalar"
        self._previous_kind = "value"
        return True

    def _initialize_scalar(self, char: str) -> bool:
        if char == "-":
            self._number_state = "sign"
            return True
        if char == "0":
            self._number_state = "zero"
            return True
        if char in "123456789":
            self._number_state = "integer"
            return True
        targets = {"t": "true", "f": "false", "n": "null"}
        target = targets.get(char)
        if target is None:
            return False
        self._literal = char
        self._literal_target = target
        return True

    def _append_scalar_character(self, char: str) -> bool:
        if self._literal_target is not None:
            candidate = self._literal + char
            if not self._literal_target.startswith(candidate):
                return False
            self._literal = candidate
            return True
        return self._advance_number(char)

    def _advance_number(self, char: str) -> bool:
        state = self._number_state
        if state == "sign":
            if char == "0":
                self._number_state = "zero"
                return True
            if char in "123456789":
                self._number_state = "integer"
                return True
            return False
        if state == "zero":
            return self._advance_number_suffix(char)
        if state == "integer":
            if char in "0123456789":
                return True
            return self._advance_number_suffix(char)
        if state == "decimal_point":
            if char not in "0123456789":
                return False
            self._number_state = "fraction"
            return True
        if state == "fraction":
            if char in "0123456789":
                return True
            if char in "eE":
                self._number_state = "exponent_mark"
                return True
            return False
        if state == "exponent_mark":
            if char in "+-":
                self._number_state = "exponent_sign"
                return True
            if char in "0123456789":
                self._number_state = "exponent"
                return True
            return False
        if state == "exponent_sign":
            if char not in "0123456789":
                return False
            self._number_state = "exponent"
            return True
        if state == "exponent":
            return char in "0123456789"
        return False

    def _advance_number_suffix(self, char: str) -> bool:
        if char == ".":
            self._number_state = "decimal_point"
            return True
        if char in "eE":
            self._number_state = "exponent_mark"
            return True
        return False

    def _scalar_complete(self) -> bool:
        if self._literal_target is not None:
            return self._literal == self._literal_target
        return self._number_state in {"zero", "integer", "fraction", "exponent"}

    def _consume_key(self) -> bool:
        if not self._frames:
            return False
        frame = self._frames[-1]
        if frame.opener != "{" or frame.expectation != "key_or_end":
            return False
        frame.expectation = "colon"
        frame.after_comma = False
        return True

    def _consume_value(self) -> bool:
        if not self._root_started:
            self._root_started = True
            return True
        if not self._frames:
            return False
        frame = self._frames[-1]
        if frame.expectation not in {"value", "value_or_end"}:
            return False
        frame.expectation = "comma_or_end"
        frame.after_comma = False
        return True

    def _emit_value_start(self) -> None:
        if self._pending_line:
            self._result.extend(("\n", " " * (self._depth * self.indent)))
            self._formatted_dirty = True
        self._pending_line = False


def format_json_prefix(text: str, indent: int = 2) -> str | None:
    """Pretty-format a valid JSON object or array prefix without adding content."""
    formatter = JsonPrefixFormatter(indent=indent)
    return formatter.append(text)
