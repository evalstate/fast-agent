from __future__ import annotations

from pathlib import Path

import click
import typer

from fast_agent.cli.command_support import (
    ensure_context_object,
    resolve_context_path_option,
    resolve_context_string_option,
)


def _context(obj: object | None = None) -> typer.Context:
    ctx = typer.Context(click.Command("test"))
    ctx.obj = obj
    return ctx


def test_resolve_context_string_option_trims_command_value() -> None:
    ctx = _context({"registry": "fallback"})

    assert (
        resolve_context_string_option(ctx, key="registry", command_value=" https://example.test ")
        == "https://example.test"
    )


def test_resolve_context_string_option_uses_trimmed_context_value() -> None:
    ctx = _context({"registry": " ./marketplace.json "})

    assert resolve_context_string_option(ctx, key="registry") == "./marketplace.json"


def test_resolve_context_string_option_ignores_blank_values() -> None:
    ctx = _context({"registry": "   "})

    assert resolve_context_string_option(ctx, key="registry", command_value=" ") is None


def test_resolve_context_path_option_uses_trimmed_context_value() -> None:
    ctx = _context({"skills_dir": " ./skills "})

    assert resolve_context_path_option(ctx, key="skills_dir") == Path("./skills")


def test_ensure_context_object_preserves_existing_dict() -> None:
    obj: dict[str, object] = {"registry": "one"}
    ctx = _context(obj)

    assert ensure_context_object(ctx) is obj
