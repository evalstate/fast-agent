from __future__ import annotations

import base64
import mimetypes
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from mcp_types import BlobResourceContents, EmbeddedResource, ImageContent, TextContent
from rich.console import Group, RenderableType
from rich.text import Text

from fast_agent.core.logging.logger import get_logger
from fast_agent.ui.terminal_images.probe import prepare_terminal_image_probe

if TYPE_CHECKING:
    from fast_agent.command_actions.models import PluginCommandActionImage
    from fast_agent.config import Settings, TerminalImageSettings
    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

logger = get_logger(__name__)
MAX_TERMINAL_IMAGE_SOURCE_BYTES = 25 * 1024 * 1024
TERMINAL_IMAGE_FETCH_TIMEOUT_SECONDS = 10.0

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
    width: int | str | None = None
    height: int | str | None = None


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
    if settings is None or content is None or not settings.enabled or settings.backend == "none":
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
    if settings is None or content is None or not settings.enabled or settings.backend == "none":
        return None
    return render_image_items(settings, extract_image_render_items(content))


def render_plugin_command_images_for_settings(
    settings: TerminalImageSettings | None,
    images: Sequence[PluginCommandActionImage],
    *,
    base_dir: Path | None = None,
) -> RenderableType | None:
    if settings is None or not images or not settings.enabled or settings.backend == "none":
        return None
    items = [
        ImageRenderItem(artifact)
        for image in images
        if (artifact := _artifact_from_plugin_image(image, base_dir=base_dir)) is not None
    ]
    return render_image_items(settings, items)


def render_image_items(
    settings: TerminalImageSettings,
    items: Sequence[ImageRenderItem],
) -> Group | None:
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
    return _active_terminal_image_settings(config)


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
            mime_type=item.mime_type,
            label=_label(index, item.mime_type, len(data), None),
        )

    if isinstance(item, EmbeddedResource) and isinstance(item.resource, BlobResourceContents):
        mime_type = item.resource.mime_type or "application/octet-stream"
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


def _artifact_from_plugin_image(
    image: PluginCommandActionImage,
    *,
    base_dir: Path | None,
) -> ImageArtifact | None:
    source = image.source.strip()
    if not source:
        return None

    data, mime_type = _read_image_source(
        source,
        base_dir=base_dir,
        mime_type=image.mime_type,
    )
    if data is None or mime_type is None:
        return None
    label = image.label or source
    return ImageArtifact(
        data=data,
        mime_type=mime_type,
        label=f"[IMAGE: {label}, {mime_type}, {len(data)} bytes]",
        uri=source,
        width=image.width,
        height=image.height,
    )


def _read_image_source(
    source: str,
    *,
    base_dir: Path | None,
    mime_type: str | None,
) -> tuple[bytes | None, str | None]:
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        return _read_image_url(source, mime_type=mime_type)
    if parsed.scheme == "file":
        return _read_image_path(Path(unquote(parsed.path)), mime_type=mime_type)
    if parsed.scheme:
        return None, None
    path = Path(source).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return _read_image_path(path, mime_type=mime_type)


def _read_image_url(
    url: str,
    *,
    mime_type: str | None,
) -> tuple[bytes | None, str | None]:
    try:
        request = Request(url, headers={"User-Agent": "fast-agent"})
        with urlopen(request, timeout=TERMINAL_IMAGE_FETCH_TIMEOUT_SECONDS) as response:
            header_mime = response.headers.get_content_type()
            data = response.read(MAX_TERMINAL_IMAGE_SOURCE_BYTES + 1)
    except Exception:
        logger.debug("Failed to fetch terminal image URL", url=url)
        return None, None

    if len(data) > MAX_TERMINAL_IMAGE_SOURCE_BYTES:
        logger.debug("Terminal image URL exceeded size limit", url=url)
        return None, None

    resolved_mime = _resolve_image_mime(mime_type or header_mime, url)
    if resolved_mime is None:
        return None, None
    return data, resolved_mime


def _read_image_path(
    path: Path,
    *,
    mime_type: str | None,
) -> tuple[bytes | None, str | None]:
    try:
        resolved = path.expanduser().resolve()
        if not resolved.is_file() or resolved.stat().st_size > MAX_TERMINAL_IMAGE_SOURCE_BYTES:
            return None, None
        data = resolved.read_bytes()
    except Exception:
        logger.debug("Failed to read terminal image path", path=str(path))
        return None, None

    resolved_mime = _resolve_image_mime(mime_type, resolved.as_posix())
    if resolved_mime is None:
        return None, None
    return data, resolved_mime


def _resolve_image_mime(mime_type: str | None, source: str) -> str | None:
    guessed = mime_type or mimetypes.guess_type(source)[0]
    if guessed and guessed.startswith("image/"):
        return guessed
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


_DEFAULT_AUTO_WIDTH_FRACTION = 0.8
_DEFAULT_AUTO_HEIGHT_FRACTION = 0.6
_FALLBACK_CELL_WIDTH_PX = 10.0
_FALLBACK_CELL_HEIGHT_PX = 20.0


def _read_image_pixel_size(data: bytes) -> tuple[int, int] | None:
    try:
        from PIL import Image as PILImage

        with PILImage.open(BytesIO(data)) as image:
            width, height = image.size
    except Exception:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def _terminal_viewport_geometry() -> tuple[int, int, float, float]:
    """Return (cols, rows, cell_width_px, cell_height_px) for the active terminal pane."""
    fallback_size = shutil.get_terminal_size((80, 24))
    cols = max(1, fallback_size.columns)
    rows = max(1, fallback_size.lines)
    cell_w = _FALLBACK_CELL_WIDTH_PX
    cell_h = _FALLBACK_CELL_HEIGHT_PX

    try:
        from textual_image import _terminal as terminal

        win_rows, win_cols, win_px_w, win_px_h = terminal.get_tiocgwinsz()
        if win_cols > 0 and win_rows > 0:
            cols = win_cols
            rows = win_rows
            if win_px_w > 0 and win_px_h > 0:
                candidate_w = win_px_w / win_cols
                candidate_h = win_px_h / win_rows
                if candidate_w > 0 and candidate_h >= candidate_w:
                    return cols, rows, candidate_w, candidate_h
    except Exception:
        pass

    try:
        from textual_image._terminal import get_cell_size

        cell_size = get_cell_size()
        if cell_size.width > 0 and cell_size.height >= cell_size.width:
            cell_w = float(cell_size.width)
            cell_h = float(cell_size.height)
    except Exception:
        pass

    return cols, rows, cell_w, cell_h


def _resolve_size_bound(
    spec: int | str | None,
    limit: int,
    *,
    auto_fraction: float,
) -> int:
    if isinstance(spec, int):
        return max(1, min(spec, limit))
    if isinstance(spec, str):
        stripped = spec.strip()
        if stripped.endswith("%"):
            try:
                pct = int(stripped[:-1])
                return max(1, min(round(limit * pct / 100), limit))
            except ValueError:
                pass
    return max(1, round(limit * auto_fraction))


def _resolve_render_dimensions(
    artifact: ImageArtifact,
    settings: TerminalImageSettings,
) -> tuple[int | str | None, int | str | None]:
    """Compute aspect-ratio-aware terminal cell dimensions without upscaling past 1:1."""
    width_spec = artifact.width if artifact.width is not None else settings.width
    height_spec = artifact.height if artifact.height is not None else settings.height

    if isinstance(width_spec, int) and isinstance(height_spec, int):
        return width_spec, height_spec

    pixel_size = _read_image_pixel_size(artifact.data)
    if pixel_size is None:
        return width_spec, height_spec

    img_w, img_h = pixel_size
    cols, rows, cell_w, cell_h = _terminal_viewport_geometry()

    settings_max_cols = _resolve_size_bound(
        settings.width, cols, auto_fraction=_DEFAULT_AUTO_WIDTH_FRACTION
    )
    settings_max_rows = _resolve_size_bound(
        settings.height, rows, auto_fraction=_DEFAULT_AUTO_HEIGHT_FRACTION
    )

    max_cols = settings_max_cols
    if artifact.width is not None:
        max_cols = min(
            max_cols,
            _resolve_size_bound(artifact.width, cols, auto_fraction=_DEFAULT_AUTO_WIDTH_FRACTION),
        )

    max_rows = settings_max_rows
    if artifact.height is not None:
        max_rows = min(
            max_rows,
            _resolve_size_bound(artifact.height, rows, auto_fraction=_DEFAULT_AUTO_HEIGHT_FRACTION),
        )

    max_px_w = max_cols * cell_w
    max_px_h = max_rows * cell_h
    scale = min(1.0, max_px_w / img_w, max_px_h / img_h)

    target_cols = max(1, min(max_cols, round((img_w * scale) / cell_w)))
    target_rows = max(1, min(max_rows, round((img_h * scale) / cell_h)))
    return target_cols, target_rows


def _render_artifact(
    artifact: ImageArtifact,
    settings: TerminalImageSettings,
) -> RenderableType | None:
    image_cls = _resolve_textual_image_class(settings.backend)
    if image_cls is None:
        return None

    width, height = _resolve_render_dimensions(artifact, settings)
    try:
        return cast(
            "RenderableType",
            image_cls(BytesIO(artifact.data), width=width, height=height),
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

    prepare_terminal_image_probe()

    if backend == "halfcell" and _herdr_active():
        try:
            module = import_module("fast_agent.ui.terminal_images.halfcell")
        except ImportError:
            logger.debug("textual-image is not installed; terminal image rendering disabled")
            return None
        return getattr(module, "HerdrAwareHalfcellImage", None)

    if backend == "sixel":
        try:
            module = import_module("fast_agent.ui.terminal_images.sixel")
        except ImportError:
            logger.debug("textual-image is not installed; terminal image rendering disabled")
            return None
        return getattr(module, "ViewportAwareSixelImage", None)

    try:
        module = import_module("textual_image.renderable")
    except ImportError:
        logger.debug("textual-image is not installed; terminal image rendering disabled")
        return None

    image_cls = getattr(module, class_name, None)
    if (
        class_name == "Image"
        and image_cls is not None
        and image_cls is getattr(module, "SixelImage", None)
    ):
        return _resolve_textual_image_class("sixel")
    return image_cls


def _herdr_active() -> bool:
    return os.environ.get("HERDR_ENV") == "1"
