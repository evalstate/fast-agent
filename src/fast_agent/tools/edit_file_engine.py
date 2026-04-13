from __future__ import annotations

import difflib
import errno
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict, TypeGuard

if TYPE_CHECKING:
    from collections.abc import Sequence

EditFileErrorCode = Literal[
    "file_not_found",
    "is_directory",
    "no_match",
    "multiple_matches",
    "no_op",
    "permission_denied",
    "empty_old_string",
    "encoding_error",
]


class MatchLocation(TypedDict):
    line_start: int
    line_end: int


class EditFileSuccess(TypedDict):
    success: Literal[True]
    path: str
    line_start: int
    line_end: int
    replacements: int
    diff: str


class EditFileError(TypedDict, total=False):
    success: Literal[False]
    error: EditFileErrorCode
    message: str
    path: str
    matches: list[MatchLocation]


type EditFileResult = EditFileSuccess | EditFileError


_PERMISSION_ERRNOS: Final = {errno.EACCES, errno.EPERM}


def serialize_edit_file_result(result: EditFileResult) -> dict[str, Any]:
    if _is_edit_file_error(result):
        return _serialize_edit_file_error(result)
    if _is_edit_file_success(result):
        return _serialize_edit_file_success(result)
    raise AssertionError("Unreachable edit_file result variant")


def _is_edit_file_error(result: EditFileResult) -> TypeGuard[EditFileError]:
    return "error" in result


def _is_edit_file_success(result: EditFileResult) -> TypeGuard[EditFileSuccess]:
    return result["success"] is True


def _serialize_edit_file_success(result: EditFileSuccess) -> dict[str, Any]:
    return {
        "success": True,
        "path": result["path"],
        "line_start": result["line_start"],
        "line_end": result["line_end"],
        "replacements": result["replacements"],
        "diff": result["diff"],
    }


def _serialize_edit_file_error(result: EditFileError) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "success": False,
        "error": result["error"],
        "message": result["message"],
        "path": result["path"],
    }
    if "matches" in result:
        payload["matches"] = result["matches"]
    return payload


def edit_file(
    path: Path,
    *,
    display_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> EditFileResult:
    if not path.exists():
        return _error(
            display_path,
            "file_not_found",
            f"File not found: {display_path}. Check the path, or use write_text_file to create it.",
        )
    if path.is_dir():
        return _error(
            display_path,
            "is_directory",
            f"Path is a directory, not a file: {display_path}. Use the correct file path.",
        )
    if old_string == "":
        return _error(
            display_path,
            "empty_old_string",
            "Empty search string is not allowed.",
        )
    if old_string == new_string:
        return _error(
            display_path,
            "no_op",
            "old_string and new_string are identical. No change is needed.",
        )

    read_result = _read_text_file(path, display_path=display_path)
    if not isinstance(read_result, str):
        return read_result
    original_contents = read_result

    matches = _find_match_spans(original_contents, old_string)
    if not matches:
        return _error(
            display_path,
            "no_match",
            "old_string was not found in the file. Re-read the file and use the exact current text.",
        )
    if len(matches) > 1 and not replace_all:
        return {
            "success": False,
            "error": "multiple_matches",
            "message": (
                f"Found {len(matches)} matches for old_string in {display_path}. "
                "Use replace_all=True to replace all, or provide more surrounding context "
                "to uniquely identify the target."
            ),
            "path": display_path,
            "matches": [_location_for_span(original_contents, start, end) for start, end in matches],
        }

    selected_matches = matches if replace_all else matches[:1]
    new_contents, updated_spans = _apply_matches(
        original_contents,
        selected_matches,
        new_string=new_string,
    )
    diff = _unified_diff(display_path, original_contents, new_contents)

    write_error = _write_text_file_atomic(path, new_contents)
    if write_error is not None:
        return _error(display_path, write_error, f"Permission denied for file: {display_path}.")

    line_start, line_end = _line_range_for_span(
        new_contents,
        updated_spans[0][0],
        updated_spans[-1][1],
    )
    return {
        "success": True,
        "path": display_path,
        "line_start": line_start,
        "line_end": line_end,
        "replacements": len(selected_matches),
        "diff": diff,
    }


def _read_text_file(path: Path, *, display_path: str) -> str | EditFileError:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return handle.read()
    except UnicodeDecodeError:
        return _error(
            display_path,
            "encoding_error",
            f"File could not be decoded as UTF-8: {display_path}. Check file encoding.",
        )
    except PermissionError:
        return _error(
            display_path,
            "permission_denied",
            f"Permission denied for file: {display_path}.",
        )
    except OSError as exc:
        if exc.errno in _PERMISSION_ERRNOS:
            return _error(
                display_path,
                "permission_denied",
                f"Permission denied for file: {display_path}.",
            )
        raise


def _write_text_file_atomic(path: Path, contents: str) -> EditFileErrorCode | None:
    temp_path: Path | None = None
    try:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.close(fd)
        temp_path = Path(temp_name)
        try:
            temp_path.chmod(path.stat().st_mode)
        except OSError:
            pass
        with temp_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(contents)
        os.replace(temp_path, path)
    except PermissionError:
        return "permission_denied"
    except OSError as exc:
        if exc.errno in _PERMISSION_ERRNOS:
            return "permission_denied"
        raise
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)
    return None


def _find_match_spans(contents: str, old_string: str) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    start = 0
    while True:
        index = contents.find(old_string, start)
        if index < 0:
            return matches
        end = index + len(old_string)
        matches.append((index, end))
        start = end


def _apply_matches(
    contents: str,
    matches: Sequence[tuple[int, int]],
    *,
    new_string: str,
) -> tuple[str, list[tuple[int, int]]]:
    pieces: list[str] = []
    updated_spans: list[tuple[int, int]] = []
    source_index = 0
    output_index = 0

    for start, end in matches:
        unchanged = contents[source_index:start]
        pieces.append(unchanged)
        output_index += len(unchanged)

        replacement_start = output_index
        pieces.append(new_string)
        output_index += len(new_string)
        updated_spans.append((replacement_start, output_index))

        source_index = end

    tail = contents[source_index:]
    pieces.append(tail)
    return "".join(pieces), updated_spans


def _location_for_span(contents: str, start: int, end: int) -> MatchLocation:
    line_start, line_end = _line_range_for_span(contents, start, end)
    return {
        "line_start": line_start,
        "line_end": line_end,
    }


def _line_number_for_offset(contents: str, offset: int) -> int:
    return contents.count("\n", 0, offset) + 1


def _line_range_for_span(contents: str, start: int, end: int) -> tuple[int, int]:
    line_start = _line_number_for_offset(contents, start)
    if end <= start:
        return line_start, line_start

    line_end_offset = end - 1
    return line_start, _line_number_for_offset(contents, line_end_offset)


def _unified_diff(display_path: str, original_contents: str, new_contents: str) -> str:
    return "".join(
        difflib.unified_diff(
            original_contents.splitlines(keepends=True),
            new_contents.splitlines(keepends=True),
            fromfile=f"a/{display_path}",
            tofile=f"b/{display_path}",
        )
    )


def _error(path: str, code: EditFileErrorCode, message: str) -> EditFileError:
    return {
        "success": False,
        "error": code,
        "message": message,
        "path": path,
    }
