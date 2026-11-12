"""Shipped code must not let text I/O fall back to the locale encoding.

Before Python 3.15, text-mode pipes and files default to the locale code page on
Windows (for example cp1252 or cp936), which corrupts or rejects non-ASCII text.
"""

import ast
from pathlib import Path

import pytest

import fast_agent

_PACKAGE_ROOT = Path(fast_agent.__file__).parent
_SUBPROCESS_CALLS = frozenset({"run", "Popen", "check_output", "check_call", "call"})


def _keywords(call: ast.Call) -> set[str | None]:
    return {keyword.arg for keyword in call.keywords}


def _text_mode(node: ast.expr | None) -> bool:
    if node is None:
        return True
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and set(node.value) <= set("rwaxt+")
    )


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)


def _locale_dependent(call: ast.Call) -> bool:
    keywords = _keywords(call)
    if "encoding" in keywords or None in keywords:
        return False
    func = call.func
    if isinstance(func, ast.Name):
        return func.id == "open" and _text_mode(
            call.args[1] if len(call.args) > 1 else _keyword(call, "mode")
        )
    if not isinstance(func, ast.Attribute):
        return False
    owner = ast.unparse(func.value)
    match func.attr:
        case name if name in _SUBPROCESS_CALLS and owner == "subprocess":
            return bool(keywords & {"text", "universal_newlines"})
        case "fdopen" if owner == "os":
            return _text_mode(call.args[1] if len(call.args) > 1 else _keyword(call, "mode"))
        case "open":
            # Path.open("w"); other .open(...) methods take no string mode.
            mode = call.args[0] if call.args else _keyword(call, "mode")
            return isinstance(mode, ast.Constant) and _text_mode(mode)
        case "read_text":
            return not call.args
        case "write_text":
            return len(call.args) == 1 and keywords <= {"newline", "errors"}
    return False


def _shipped_sources() -> list[Path]:
    generated = _PACKAGE_ROOT / "resources"
    return [path for path in _PACKAGE_ROOT.rglob("*.py") if generated not in path.parents]


@pytest.mark.unit
def test_shipped_text_io_declares_an_encoding() -> None:
    offenders = [
        f"{path.relative_to(_PACKAGE_ROOT.parent)}:{node.lineno}: {ast.unparse(node.func)}"
        for path in _shipped_sources()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, ast.Call) and _locale_dependent(node)
    ]

    assert offenders == []
