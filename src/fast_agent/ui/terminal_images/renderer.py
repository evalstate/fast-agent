from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

from mcp.types import BlobResourceContents, EmbeddedResource, ImageContent, TextContent
from rich.console import Group, RenderableType
from rich.text import Text

from fast_agent.core.logging.logger import get_logger

if TYPE_CHECKING:
    from fast_agent.config import Settings, TerminalImageSettings
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

logger = get_logger(__name__)

_TEXTUAL_IMAGE_CLASS_BY_BACKEND: dict[str, str] = {
    "auto": "Image",
    "textual-image": "Image",
    "kitty": "TGPImage",
    "sixel": "SixelImage",
    "halfcell": "HalfcellImage",
    "unicode": "UnicodeImage",
}


@dataclass(frozen=True)
class ImageArtifact:
    data: bytes
    mime_type: str
    label: str
    uri: str | None = None


@dataclass(frozen=True)
class ImageRenderItem:
    artifact: ImageArtifact
    metadata: tuple[str, ...] = ()


def extract_image_artifacts(content: Sequence[object]) -> list[ImageArtifact]:
    """Extract renderable image payloads from MCP-style content blocks."""
    return [item.artifact for item in extract_image_render_items(content)]


def extract_image_render_items(content: Sequence[object]) -> list[ImageRenderItem]:
    """Extract image payloads and adjacent display metadata from MCP-style blocks."""
    items: list[ImageRenderItem] = []
    metadata: list[str] = []
    for block in content:
        artifact = _artifact_from_content(block, len(items) + 1)
        if artifact is not None:
            items.append(ImageRenderItem(artifact=artifact, metadata=()))
            continue
        if isinstance(block, TextContent):
            text = block.text.strip()
            if text:
                metadata.append(text)

    if not items or not metadata:
        return items

    # MCP image tools commonly return one or more image blocks followed by
    # useful details such as Image URL, file path, or seed. Show that once
    # underneath the final image item rather than losing it in the tool loop.
    last = items[-1]
    items[-1] = ImageRenderItem(last.artifact, tuple(metadata))
    return items


def render_assistant_images(
    config: Settings | None,
    content: Sequence[object] | PromptMessageExtended | None,
) -> RenderableType | None:
    settings = _assistant_settings(config)
    return render_assistant_images_for_settings(settings, content)


def render_assistant_images_for_settings(
    settings: TerminalImageSettings | None,
    content: Sequence[object] | PromptMessageExtended | None,
) -> RenderableType | None:
    if settings is None or content is None:
        return None

    blocks: Sequence[object]
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes, bytearray)):
        blocks = cast("PromptMessageExtended", content).content
    else:
        blocks = content

    return render_image_items(settings, extract_image_render_items(blocks))


def render_tool_result_images(
    config: Settings | None,
    content: Sequence[object] | None,
) -> RenderableType | None:
    settings = _tool_result_settings(config)
    return render_tool_result_images_for_settings(settings, content)


def render_tool_result_images_for_settings(
    settings: TerminalImageSettings | None,
    content: Sequence[object] | None,
) -> RenderableType | None:
    if settings is None or content is None:
        return None
    return render_image_items(settings, extract_image_render_items(content))


def render_image_items(
    settings: TerminalImageSettings,
    items: Sequence[ImageRenderItem],
) -> RenderableType | None:
    if not items:
        return None

    renderables: list[RenderableType] = []
    for item in items:
        renderable = _render_artifact(item.artifact, settings)
        if renderable is None:
            continue
        renderables.append(Text(item.artifact.label, style="dim"))
        renderables.append(renderable)
        renderables.extend(Text(metadata, style="dim") for metadata in item.metadata)

    if not renderables:
        return None
    return Group(*renderables)


def _assistant_settings(config: Settings | None) -> TerminalImageSettings | None:
    terminal_images = _active_terminal_image_settings(config)
    if terminal_images is None:
        return None
    if not terminal_images.render_assistant:
        return None
    return terminal_images


def _tool_result_settings(config: Settings | None) -> TerminalImageSettings | None:
    terminal_images = _active_terminal_image_settings(config)
    if terminal_images is None:
        return None
    # Tool-result images are rendered at the tool-result boundary. Keep
    # render_assistant as the user-facing "show generated images" switch.
    if not terminal_images.render_assistant:
        return None
    return terminal_images


def _active_terminal_image_settings(
    config: Settings | None,
) -> TerminalImageSettings | None:
    if config is None:
        return None
    terminal_images = config.logger.terminal_images
    if not terminal_images.enabled or terminal_images.backend == "none":
        return None
    return terminal_images


def _artifact_from_content(item: object, index: int) -> ImageArtifact | None:
    if isinstance(item, ImageContent):
        data = _decode_base64(item.data)
        if data is None:
            return None
        return ImageArtifact(
            data=data,
            mime_type=item.mimeType,
            label=_label(index, item.mimeType, len(data), None),
        )

    if isinstance(item, EmbeddedResource) and isinstance(item.resource, BlobResourceContents):
        mime_type = item.resource.mimeType or "application/octet-stream"
        if not mime_type.startswith("image/"):
            return None
        data = _decode_base64(item.resource.blob)
        if data is None:
            return None
        uri = str(item.resource.uri)
        return ImageArtifact(
            data=data,
            mime_type=mime_type,
            label=_label(index, mime_type, len(data), uri),
            uri=uri,
        )

    return None


def _decode_base64(data: str) -> bytes | None:
    try:
        return base64.b64decode(data, validate=True)
    except Exception:
        logger.debug("Failed to decode terminal image payload")
        return None


def _label(index: int, mime_type: str, size: int, uri: str | None) -> str:
    suffix = f", {uri}" if uri else ""
    return f"[IMAGE {index}: {mime_type}, {size} bytes{suffix}]"


def _render_artifact(
    artifact: ImageArtifact,
    settings: TerminalImageSettings,
) -> RenderableType | None:
    image_cls = _resolve_textual_image_class(settings.backend)
    if image_cls is None:
        return None

    try:
        return cast(
            "RenderableType",
            image_cls(BytesIO(artifact.data), width=settings.width, height=settings.height),
        )
    except Exception:
        logger.debug(
            "Failed to render terminal image",
            mime_type=artifact.mime_type,
            backend=settings.backend,
        )
        return None


def _resolve_textual_image_class(backend: str) -> Any | None:
    class_name = _TEXTUAL_IMAGE_CLASS_BY_BACKEND.get(backend)
    if class_name is None:
        return None

    try:
        module = import_module("textual_image.renderable")
    except ImportError:
        logger.debug("textual-image is not installed; terminal image rendering disabled")
        return None
    return getattr(module, class_name, None)
