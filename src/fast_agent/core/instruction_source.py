"""Shared helpers for resolving instruction text sources."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fast_agent.core.direct_decorators import _resolve_instruction
from fast_agent.io.source_resolver import materialized_text_source

if TYPE_CHECKING:
    from pathlib import Path


def resolve_instruction_source(source: str | Path) -> str:
    """Resolve an instruction from a path, file URI, HTTP(S) URL, or hf:// URI."""
    with materialized_text_source(source, label="instruction") as path:
        return _resolve_instruction(path)
