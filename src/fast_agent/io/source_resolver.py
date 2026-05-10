from __future__ import annotations

import tempfile
from pathlib import Path
from typing import BinaryIO, Protocol
from urllib.parse import ParseResult, urlparse

import requests

from fast_agent.io.path_uri import file_uri_to_path

REMOTE_TEXT_SCHEMES = frozenset({"http", "https", "hf"})


class HfTextFileSystem(Protocol):
    def open(self, path: str, mode: str = "rb") -> BinaryIO: ...


def read_text_source(source: str | Path, *, label: str = "source") -> str:
    """Read UTF-8 text from a filesystem path, file URI, HTTP(S) URL, or hf:// URI."""
    source_text = str(source)
    parsed = urlparse(source_text)

    if parsed.scheme in {"http", "https"}:
        try:
            response = requests.get(source_text, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ValueError(f"Could not read {label} {source_text}: {exc}") from exc
        response.encoding = response.encoding or "utf-8"
        return response.text

    if parsed.scheme == "hf":
        return _read_hf_text_source(source_text, label=label)

    path = _path_from_source(source_text, parsed)
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read {label} {source_text}: {exc}") from exc


def materialize_text_source(
    source: str | Path,
    *,
    label: str = "source",
    suffix: str | None = None,
) -> Path:
    """Return a local path for a filesystem path, file URI, or downloaded remote text URI."""
    source_text = str(source)
    parsed = urlparse(source_text)
    if parsed.scheme not in REMOTE_TEXT_SCHEMES:
        return _path_from_source(source_text, parsed).expanduser()

    text = read_text_source(source_text, label=label)
    resolved_suffix = suffix or Path(parsed.path).suffix or ".txt"
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=resolved_suffix,
        prefix="fast-agent-",
        delete=False,
        encoding="utf-8",
    ) as temp_file:
        temp_file.write(text)
        return Path(temp_file.name)


def _path_from_source(source: str, parsed: ParseResult) -> Path:
    if parsed.scheme == "file":
        return file_uri_to_path(parsed)
    return Path(source).expanduser()


def _read_hf_text_source(
    source: str,
    *,
    label: str,
    filesystem: HfTextFileSystem | None = None,
) -> str:
    try:
        fs = filesystem if filesystem is not None else _default_hf_filesystem()
        with fs.open(source, "rb") as handle:
            return handle.read().decode("utf-8")
    except UnicodeDecodeError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read {label} {source}: {exc}") from exc


def _default_hf_filesystem() -> HfTextFileSystem:
    try:
        from huggingface_hub import HfFileSystem
    except Exception as exc:
        raise ValueError("huggingface_hub is not available") from exc

    return HfFileSystem()
