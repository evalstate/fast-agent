"""
Parser for the openai/codex apply_patch port (Apache 2.0).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, TypeAlias

from fast_agent.patch.errors import InvalidHunkError, InvalidPatchError

BEGIN_PATCH_MARKER = "*** Begin Patch"
END_PATCH_MARKER = "*** End Patch"
ADD_FILE_MARKER = "*** Add File: "
DELETE_FILE_MARKER = "*** Delete File: "
UPDATE_FILE_MARKER = "*** Update File: "
MOVE_TO_MARKER = "*** Move to: "
EOF_MARKER = "*** End of File"
CHANGE_CONTEXT_MARKER = "@@ "
EMPTY_CHANGE_CONTEXT_MARKER = "@@"

PARSE_IN_STRICT_MODE = False


@dataclass
class UpdateFileChunk:
    change_context: str | None
    old_lines: list[str]
    new_lines: list[str]
    is_end_of_file: bool


@dataclass(frozen=True)
class AddFileHunk:
    kind: Literal["add"]
    path: Path
    contents: str


@dataclass(frozen=True)
class DeleteFileHunk:
    kind: Literal["delete"]
    path: Path


@dataclass(frozen=True)
class UpdateFileHunk:
    kind: Literal["update"]
    path: Path
    move_path: Path | None
    chunks: list[UpdateFileChunk]


Hunk: TypeAlias = AddFileHunk | DeleteFileHunk | UpdateFileHunk


@dataclass(frozen=True)
class ApplyPatchArgs:
    patch: str
    hunks: list[Hunk]
    workdir: str | None


class ParseMode(Enum):
    STRICT = "strict"
    LENIENT = "lenient"


def parse_patch(patch: str) -> ApplyPatchArgs:
    mode = ParseMode.STRICT if PARSE_IN_STRICT_MODE else ParseMode.LENIENT
    return parse_patch_text(patch, mode)


def parse_patch_text(patch: str, mode: ParseMode) -> ApplyPatchArgs:
    lines = patch.strip().splitlines()
    lines = _check_patch_boundaries(lines, mode)

    hunks: list[Hunk] = []
    last_line_index = len(lines) - 1
    remaining_lines = lines[1:last_line_index]
    line_number = 2
    while remaining_lines:
        hunk, hunk_lines = parse_one_hunk(remaining_lines, line_number)
        hunks.append(hunk)
        line_number += hunk_lines
        remaining_lines = remaining_lines[hunk_lines:]

    patch_text = "\n".join(lines)
    return ApplyPatchArgs(patch=patch_text, hunks=hunks, workdir=None)


def _check_patch_boundaries(lines: list[str], mode: ParseMode) -> list[str]:
    try:
        _check_patch_boundaries_strict(lines)
        return lines
    except InvalidPatchError as exc:
        if mode is ParseMode.STRICT:
            raise
        return _check_patch_boundaries_lenient(lines, exc)


def _check_patch_boundaries_strict(lines: list[str]) -> None:
    first_line = lines[0].strip() if lines else None
    last_line = lines[-1].strip() if lines else None
    _check_start_and_end_lines_strict(first_line, last_line)


def _check_patch_boundaries_lenient(
    lines: list[str],
    original_error: InvalidPatchError,
) -> list[str]:
    if not lines:
        raise original_error
    if len(lines) < 4:
        raise original_error
    first = lines[0]
    last = lines[-1]
    if (
        first in {"<<EOF", "<<'EOF'", '<<"EOF"'}
        and last.endswith("EOF")
        and len(lines) >= 4
    ):
        inner_lines = lines[1:-1]
        _check_patch_boundaries_strict(inner_lines)
        return inner_lines
    raise original_error


def _check_start_and_end_lines_strict(
    first_line: str | None,
    last_line: str | None,
) -> None:
    if first_line == BEGIN_PATCH_MARKER and last_line == END_PATCH_MARKER:
        return
    if first_line != BEGIN_PATCH_MARKER:
        raise InvalidPatchError("The first line of the patch must be '*** Begin Patch'")
    raise InvalidPatchError("The last line of the patch must be '*** End Patch'")


def parse_one_hunk(lines: list[str], line_number: int) -> tuple[Hunk, int]:
    first_line = lines[0].strip()
    path = _strip_prefix(first_line, ADD_FILE_MARKER)
    if path is not None:
        contents = ""
        parsed_lines = 1
        for add_line in lines[1:]:
            line_to_add = _strip_prefix(add_line, "+")
            if line_to_add is None:
                break
            contents += f"{line_to_add}\n"
            parsed_lines += 1
        return AddFileHunk(kind="add", path=Path(path), contents=contents), parsed_lines
    path = _strip_prefix(first_line, DELETE_FILE_MARKER)
    if path is not None:
        return DeleteFileHunk(kind="delete", path=Path(path)), 1
    path = _strip_prefix(first_line, UPDATE_FILE_MARKER)
    if path is not None:
        remaining_lines = lines[1:]
        parsed_lines = 1

        move_path = None
        if remaining_lines:
            move_path_text = _strip_prefix(remaining_lines[0], MOVE_TO_MARKER)
            if move_path_text is not None:
                move_path = Path(move_path_text)
                remaining_lines = remaining_lines[1:]
                parsed_lines += 1

        chunks: list[UpdateFileChunk] = []
        while remaining_lines:
            if not remaining_lines:
                break
            if not remaining_lines[0].strip():
                parsed_lines += 1
                remaining_lines = remaining_lines[1:]
                continue
            if remaining_lines[0].startswith("***"):
                break

            chunk, chunk_lines = parse_update_file_chunk(
                remaining_lines,
                line_number + parsed_lines,
                allow_missing_context=not chunks,
            )
            chunks.append(chunk)
            parsed_lines += chunk_lines
            remaining_lines = remaining_lines[chunk_lines:]

        if not chunks:
            raise InvalidHunkError(
                message=f"Update file hunk for path '{path}' is empty",
                line_number=line_number,
            )
        return (
            UpdateFileHunk(
                kind="update",
                path=Path(path),
                move_path=move_path,
                chunks=chunks,
            ),
            parsed_lines,
        )

    raise InvalidHunkError(
        message=(
            f"'{first_line}' is not a valid hunk header. "
            "Valid hunk headers: '*** Add File: {path}', '*** Delete File: {path}', "
            "'*** Update File: {path}'"
        ),
        line_number=line_number,
    )


def parse_update_file_chunk(
    lines: list[str],
    line_number: int,
    *,
    allow_missing_context: bool,
) -> tuple[UpdateFileChunk, int]:
    if not lines:
        raise InvalidHunkError(
            message="Update hunk does not contain any lines",
            line_number=line_number,
        )

    if lines[0] == EMPTY_CHANGE_CONTEXT_MARKER:
        change_context = None
        start_index = 1
    else:
        context = _strip_prefix(lines[0], CHANGE_CONTEXT_MARKER)
        if context is not None:
            change_context = context
            start_index = 1
        else:
            if not allow_missing_context:
                raise InvalidHunkError(
                    message=(
                        "Expected update hunk to start with a @@ context marker, got: "
                        f"'{lines[0]}'"
                    ),
                    line_number=line_number,
                )
            change_context = None
            start_index = 0

    if start_index >= len(lines):
        raise InvalidHunkError(
            message="Update hunk does not contain any lines",
            line_number=line_number + 1,
        )

    chunk = UpdateFileChunk(
        change_context=change_context,
        old_lines=[],
        new_lines=[],
        is_end_of_file=False,
    )
    parsed_lines = 0
    for line in lines[start_index:]:
        if line == EOF_MARKER:
            if parsed_lines == 0:
                raise InvalidHunkError(
                    message="Update hunk does not contain any lines",
                    line_number=line_number + 1,
                )
            chunk.is_end_of_file = True
            parsed_lines += 1
            break

        if not line:
            chunk.old_lines.append("")
            chunk.new_lines.append("")
            parsed_lines += 1
            continue

        prefix = line[0]
        if prefix == " ":
            content = line[1:]
            chunk.old_lines.append(content)
            chunk.new_lines.append(content)
            parsed_lines += 1
        elif prefix == "+":
            chunk.new_lines.append(line[1:])
            parsed_lines += 1
        elif prefix == "-":
            chunk.old_lines.append(line[1:])
            parsed_lines += 1
        else:
            if parsed_lines == 0:
                raise InvalidHunkError(
                    message=(
                        "Unexpected line found in update hunk: "
                        f"'{line}'. Every line should start with ' ' (context line), "
                        "'+' (added line), or '-' (removed line)"
                    ),
                    line_number=line_number + 1,
                )
            break

    return chunk, parsed_lines + start_index


def _strip_prefix(value: str, prefix: str) -> str | None:
    if value.startswith(prefix):
        return value[len(prefix) :]
    return None
