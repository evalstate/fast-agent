from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TextIO, cast

if TYPE_CHECKING:
    from pathlib import Path

from fast_agent.patch.errors import ApplyPatchError, InvalidHunkError, InvalidPatchError, IoError
from fast_agent.patch.parser import (
    AddFileHunk,
    DeleteFileHunk,
    Hunk,
    UpdateFileChunk,
    UpdateFileHunk,
    parse_patch,
)
from fast_agent.patch.seek_sequence import seek_sequence


@dataclass(frozen=True)
class AffectedPaths:
    added: list[Path]
    modified: list[Path]
    deleted: list[Path]


@dataclass(frozen=True)
class AppliedPatch:
    original_contents: str
    new_contents: str


def apply_patch(patch: str, stdout: TextIO, stderr: TextIO) -> None:
    try:
        parsed = parse_patch(patch)
    except InvalidPatchError as exc:
        stderr.write(f"Invalid patch: {exc}\n")
        raise
    except InvalidHunkError as exc:
        stderr.write(f"Invalid patch hunk on line {exc.line_number}: {exc.message}\n")
        raise

    apply_hunks(parsed.hunks, stdout, stderr)


def apply_hunks(hunks: list[Hunk], stdout: TextIO, stderr: TextIO) -> None:
    try:
        affected = apply_hunks_to_files(hunks)
    except ApplyPatchError as exc:
        stderr.write(f"{exc}\n")
        raise

    print_summary(affected, stdout)


def apply_hunks_to_files(hunks: list[Hunk]) -> AffectedPaths:
    if not hunks:
        raise ApplyPatchError("No files were modified.")

    added: list[Path] = []
    modified: list[Path] = []
    deleted: list[Path] = []

    for hunk in hunks:
        if hunk.kind == "add":
            add_hunk = cast("AddFileHunk", hunk)
            _write_file(add_hunk.path, add_hunk.contents)
            added.append(add_hunk.path)
        elif hunk.kind == "delete":
            delete_hunk = cast("DeleteFileHunk", hunk)
            try:
                delete_hunk.path.unlink()
            except OSError as exc:
                raise ApplyPatchError(f"Failed to delete file {delete_hunk.path}") from exc
            deleted.append(delete_hunk.path)
        elif hunk.kind == "update":
            update_hunk = cast("UpdateFileHunk", hunk)
            applied = derive_new_contents_from_chunks(update_hunk.path, update_hunk.chunks)
            destination = update_hunk.move_path
            if destination is not None:
                _write_file(destination, applied.new_contents)
                try:
                    update_hunk.path.unlink()
                except OSError as exc:
                    raise ApplyPatchError(
                        f"Failed to remove original {update_hunk.path}"
                    ) from exc
                modified.append(destination)
            else:
                _write_file(update_hunk.path, applied.new_contents)
                modified.append(update_hunk.path)
        else:
            raise ApplyPatchError(f"Unsupported hunk kind: {hunk}")

    return AffectedPaths(added=added, modified=modified, deleted=deleted)


def derive_new_contents_from_chunks(path: Path, chunks: list[UpdateFileChunk]) -> AppliedPatch:
    try:
        original_contents = _read_file(path)
    except OSError as exc:
        raise IoError(
            context=f"Failed to read file to update {path}",
            source=_format_os_error(exc),
        ) from exc

    original_lines = original_contents.split("\n")
    if original_lines and original_lines[-1] == "":
        original_lines.pop()

    replacements = compute_replacements(original_lines, path, chunks)
    new_lines = apply_replacements(original_lines, replacements)
    if not new_lines or new_lines[-1] != "":
        new_lines.append("")
    new_contents = "\n".join(new_lines)
    return AppliedPatch(original_contents=original_contents, new_contents=new_contents)


def compute_replacements(
    original_lines: list[str],
    path: Path,
    chunks: list[UpdateFileChunk],
) -> list[tuple[int, int, list[str]]]:
    replacements: list[tuple[int, int, list[str]]] = []
    line_index = 0

    for chunk in chunks:
        if chunk.change_context is not None:
            found = seek_sequence(
                original_lines,
                [chunk.change_context],
                line_index,
                False,
            )
            if found is None:
                raise ApplyPatchError(
                    f"Failed to find context '{chunk.change_context}' in {path}"
                )
            line_index = found + 1

        if not chunk.old_lines:
            insertion_index = (
                len(original_lines) - 1
                if original_lines and original_lines[-1] == ""
                else len(original_lines)
            )
            replacements.append((insertion_index, 0, list(chunk.new_lines)))
            continue

        pattern = list(chunk.old_lines)
        new_slice = list(chunk.new_lines)
        found = seek_sequence(original_lines, pattern, line_index, chunk.is_end_of_file)

        if found is None and pattern and pattern[-1] == "":
            pattern = pattern[:-1]
            if new_slice and new_slice[-1] == "":
                new_slice = new_slice[:-1]
            found = seek_sequence(original_lines, pattern, line_index, chunk.is_end_of_file)

        if found is None:
            expected = "\n".join(chunk.old_lines)
            raise ApplyPatchError(f"Failed to find expected lines in {path}:\n{expected}")

        replacements.append((found, len(pattern), new_slice))
        line_index = found + len(pattern)

    replacements.sort(key=lambda item: item[0])
    return replacements


def apply_replacements(
    lines: list[str],
    replacements: list[tuple[int, int, list[str]]],
) -> list[str]:
    updated = list(lines)
    for start_index, old_len, new_segment in reversed(replacements):
        for _ in range(old_len):
            if start_index < len(updated):
                updated.pop(start_index)
        for offset, new_line in enumerate(new_segment):
            updated.insert(start_index + offset, new_line)
    return updated


def print_summary(affected: AffectedPaths, out: TextIO) -> None:
    out.write("Success. Updated the following files:\n")
    for path in affected.added:
        out.write(f"A {path}\n")
    for path in affected.modified:
        out.write(f"M {path}\n")
    for path in affected.deleted:
        out.write(f"D {path}\n")


def _write_file(path: Path, contents: str) -> None:
    _ensure_parent(path)
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(contents)
    except OSError as exc:
        raise ApplyPatchError(f"Failed to write file {path}") from exc


def _ensure_parent(path: Path) -> None:
    parent = path.parent
    if str(parent) in {"", "."}:
        return
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ApplyPatchError(f"Failed to create parent directories for {path}") from exc


def _read_file(path: Path) -> str:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.read()


def _format_os_error(error: OSError) -> str:
    if error.errno is None:
        return str(error)
    return f"{error.strerror} (os error {error.errno})"
