"""
Python port of the openai/codex apply_patch tool.

This port is derived from https://github.com/openai/codex and is licensed
under the Apache 2.0 license.
"""

from __future__ import annotations

from fast_agent.patch.engine import (
    AffectedPaths,
    apply_hunks,
    apply_hunks_to_files,
    apply_patch,
    derive_new_contents_from_chunks,
)
from fast_agent.patch.errors import ApplyPatchError, InvalidHunkError, InvalidPatchError, IoError
from fast_agent.patch.parser import (
    AddFileHunk,
    ApplyPatchArgs,
    DeleteFileHunk,
    Hunk,
    ParseMode,
    UpdateFileChunk,
    UpdateFileHunk,
    parse_patch,
    parse_patch_text,
)
from fast_agent.patch.seek_sequence import seek_sequence

__all__ = [
    "AffectedPaths",
    "AddFileHunk",
    "ApplyPatchArgs",
    "ApplyPatchError",
    "DeleteFileHunk",
    "Hunk",
    "InvalidHunkError",
    "InvalidPatchError",
    "IoError",
    "ParseMode",
    "UpdateFileChunk",
    "UpdateFileHunk",
    "apply_hunks",
    "apply_hunks_to_files",
    "apply_patch",
    "derive_new_contents_from_chunks",
    "parse_patch",
    "parse_patch_text",
    "seek_sequence",
]
