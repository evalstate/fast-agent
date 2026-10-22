"""Explain syntax errors in model-written sandbox code.

Monty rejects invalid code with its parser's first diagnostic, or, when type checking is on,
with type-checker diagnostics that bury the syntax error among errors caused by it. CPython's
parser reports the same mistakes with clearer messages, so invalid code is compiled (never run)
on the host to describe it. The commonest mistake is embedding text that contains triple quotes,
such as a heredoc or source code with docstrings, inside a string with the same delimiter.
"""

from __future__ import annotations

import ast
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

_TRIPLE_QUOTES = ('"""', "'''")


def describe_syntax_error(code: str, *, direct_functions: Sequence[str] = ()) -> str | None:
    """Return a CPython syntax error report for ``code``, or None when CPython accepts it.

    ``direct_functions`` names sandbox functions the model can also call directly as tools,
    which take embedded text without any quoting.
    """
    try:
        compile(
            code,
            "ptc.py",
            "exec",
            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT | ast.PyCF_ONLY_AST,
            dont_inherit=True,
        )
    except SyntaxError as exc:
        report = "".join(traceback.format_exception_only(exc)).rstrip()
        hint = _nested_quote_hint(exc, direct_functions)
        return f"{report}\n{hint}" if hint else report
    except (ValueError, RecursionError, MemoryError):
        return None
    return None


def _nested_quote_hint(exc: SyntaxError, direct_functions: Sequence[str]) -> str | None:
    line = exc.text or ""
    delimiter = next((quote for quote in _TRIPLE_QUOTES if quote in line), None)
    if delimiter is None:
        if "triple-quoted" not in exc.msg:
            return None
        delimiter = '"""'
    other = _TRIPLE_QUOTES[1] if delimiter == _TRIPLE_QUOTES[0] else _TRIPLE_QUOTES[0]
    hint = (
        f"Hint: a string opened with {delimiter} ends at the next {delimiter}, so text that "
        f"itself contains {delimiter} (a docstring, heredoc or embedded source) must be "
        f"delimited with {other} instead (r{other}...{other} keeps backslashes as written)."
    )
    if direct_functions:
        names = ", ".join(f"`{name}`" for name in direct_functions)
        hint += (
            f" Or make the call outside code: {names} can be called directly as tools, "
            "which take the text without quoting."
        )
    return hint
