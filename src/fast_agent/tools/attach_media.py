"""Build MCP attachment blocks for local files and provider-fetchable URIs."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import unquote, urlparse

from mcp.types import (
    BlobResourceContents,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
)
from pydantic import AnyUrl

from fast_agent.io.path_uri import file_uri_to_path
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.mime_utils import (
    guess_mime_type,
    is_image_mime_type,
    is_text_mime_type,
    normalize_mime_type,
)
from fast_agent.utils.numeric import positive_int_or_none
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from fast_agent.llm.model_info import ModelInfo

DEFAULT_ATTACH_MEDIA_MAX_BYTES = 20 * 1024 * 1024

ATTACHABLE_MIME_TYPES = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "application/pdf",
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/mp4",
    "video/mp4",
    "video/mpeg",
    "video/quicktime",
    "video/webm",
]

type SourceKind = Literal["link", "local"]


@dataclass(frozen=True, slots=True)
class AttachMediaResult:
    """Result of converting a source into an MCP content block."""

    block: ContentBlock
    source: str
    mime_type: str
    display_name: str
    linked: bool


@dataclass(frozen=True, slots=True)
class _SourceInfo:
    raw_source: str
    resolved_source: str
    kind: SourceKind
    mime_type: str
    display_name: str
    local_path: Path | None = None


def supported_attach_media_mime_types(model_info: ModelInfo | None) -> list[str]:
    """Return known attachment MIME types supported by the current model."""
    if model_info is None:
        return []

    return [
        mime_type
        for mime_type in ATTACHABLE_MIME_TYPES
        if model_info.supports_mime(mime_type, resource_source="embedded")
        or model_info.supports_mime(
            mime_type,
            resource_source="link",
        )
    ]


def model_supports_attach_media(model_info: ModelInfo | None) -> bool:
    """Return whether the model supports at least one non-text attachment MIME."""
    return bool(supported_attach_media_mime_types(model_info))


def normalize_attach_media_max_bytes(value: object) -> int:
    max_bytes = positive_int_or_none(value)
    if max_bytes is None:
        raise ValueError("attach media maximum size must be an integer greater than or equal to 1")
    return max_bytes


def build_attach_media(
    source: str,
    *,
    base_directory: Path,
    mime_type: str | None = None,
    name: str | None = None,
    description: str | None = None,
    model_info: ModelInfo | None = None,
    max_bytes: int = DEFAULT_ATTACH_MEDIA_MAX_BYTES,
) -> AttachMediaResult:
    """Create an MCP attachment block for a local file or provider-fetchable URI."""
    max_byte_limit = normalize_attach_media_max_bytes(max_bytes)
    source_info = _classify_source(
        source,
        base_directory=base_directory,
        mime_type=mime_type,
        name=name,
    )

    if is_text_mime_type(source_info.mime_type):
        raise ValueError(
            f"Error: '{source_info.mime_type}' is text content; use read_text_file for text/code files"
        )

    resource_source = "link" if source_info.kind == "link" else "embedded"
    if model_info is not None and not model_info.supports_mime(
        source_info.mime_type,
        resource_source=resource_source,
    ):
        raise ValueError(
            "Error: current model does not support "
            f"{resource_source} attachments with MIME type '{source_info.mime_type}'"
        )

    if (
        resource_source == "link"
        and model_info is not None
        and model_info.provider == Provider.GOOGLE
        and source_info.mime_type == "application/pdf"
    ):
        raise ValueError(
            "Error: Google attachments do not support arbitrary remote PDF links yet; "
            "attach a local PDF file instead"
        )

    if source_info.kind == "link":
        block = ResourceLink(
            type="resource_link",
            uri=AnyUrl(source_info.resolved_source),
            name=source_info.display_name,
            mimeType=source_info.mime_type,
            description=description,
        )
        return AttachMediaResult(
            block=block,
            source=source_info.resolved_source,
            mime_type=source_info.mime_type,
            display_name=source_info.display_name,
            linked=True,
        )

    if source_info.local_path is None:
        raise ValueError("Error: local attachment path could not be resolved")

    size = source_info.local_path.stat().st_size
    if size > max_byte_limit:
        limit_mb = max_byte_limit / (1024 * 1024)
        actual_mb = size / (1024 * 1024)
        raise ValueError(
            f"Error: attachment is {actual_mb:.1f} MB; maximum inline attachment size is "
            f"{limit_mb:.1f} MB"
        )

    data = source_info.local_path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    if is_image_mime_type(source_info.mime_type):
        block = ImageContent(type="image", data=encoded, mimeType=source_info.mime_type)
    else:
        block = EmbeddedResource(
            type="resource",
            resource=BlobResourceContents(
                uri=AnyUrl(source_info.resolved_source),
                blob=encoded,
                mimeType=source_info.mime_type,
            ),
        )

    return AttachMediaResult(
        block=block,
        source=source_info.resolved_source,
        mime_type=source_info.mime_type,
        display_name=source_info.display_name,
        linked=False,
    )


def _classify_source(
    source: str,
    *,
    base_directory: Path,
    mime_type: str | None,
    name: str | None,
) -> _SourceInfo:
    raw_source = source.strip()
    if not raw_source:
        raise ValueError("Error: 'source' argument is required and must be a non-empty string")

    parsed = urlparse(raw_source)
    normalized_mime = normalize_mime_type(mime_type) if mime_type else None

    if parsed.scheme == "file":
        local_path = file_uri_to_path(parsed).expanduser()
        if not local_path.is_absolute():
            local_path = (base_directory / local_path).resolve()
        else:
            local_path = local_path.resolve()
        return _local_source_info(raw_source, local_path, normalized_mime, name)

    if parsed.scheme in {"http", "https", "gs"} or _is_gemini_file_uri(raw_source):
        inferred_mime = normalized_mime or _infer_remote_mime(raw_source)
        display_name = name or _remote_display_name(raw_source)
        return _SourceInfo(
            raw_source=raw_source,
            resolved_source=raw_source,
            kind="link",
            mime_type=inferred_mime,
            display_name=display_name,
        )

    if parsed.scheme:
        if parsed.scheme == "internal":
            raise ValueError(
                "Error: attach_media does not read internal resources; use get_resource for "
                "internal:// or MCP resource URIs"
            )
        raise ValueError(
            f"Error: unsupported attachment URI scheme '{parsed.scheme}'; use get_resource for "
            "internal:// or MCP resource URIs"
        )

    local_path = Path(raw_source).expanduser()
    if not local_path.is_absolute():
        local_path = (base_directory / local_path).resolve()
    else:
        local_path = local_path.resolve()
    return _local_source_info(raw_source, local_path, normalized_mime, name)


def _local_source_info(
    raw_source: str,
    local_path: Path,
    mime_type: str | None,
    name: str | None,
) -> _SourceInfo:
    if not local_path.exists():
        raise ValueError(f"Error: local attachment does not exist: {local_path}")
    if not local_path.is_file():
        raise ValueError(f"Error: local attachment is not a file: {local_path}")

    inferred_mime = mime_type or normalize_mime_type(guess_mime_type(str(local_path)))
    if inferred_mime is None:
        inferred_mime = "application/octet-stream"

    return _SourceInfo(
        raw_source=raw_source,
        resolved_source=local_path.as_uri(),
        kind="local",
        mime_type=inferred_mime,
        display_name=name or local_path.name,
        local_path=local_path,
    )


def _infer_remote_mime(source: str) -> str:
    if _is_youtube_url(source):
        return "video/mp4"

    parsed = urlparse(source)
    path_mime = normalize_mime_type(guess_mime_type(parsed.path))
    if path_mime and path_mime != "application/octet-stream":
        return path_mime
    return "application/octet-stream"


def _remote_display_name(source: str) -> str:
    parsed = urlparse(source)
    path_name = Path(unquote(parsed.path)).name
    if path_name:
        return path_name
    return parsed.netloc or source


def _is_gemini_file_uri(source: str) -> bool:
    return source.startswith("https://generativelanguage.googleapis.com/")


def _is_youtube_url(source: str) -> bool:
    parsed = urlparse(source)
    host = strip_casefold(parsed.netloc)
    return host in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"} or host.endswith(
        ".youtube.com"
    )

