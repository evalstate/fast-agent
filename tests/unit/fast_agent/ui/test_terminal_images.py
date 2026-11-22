import base64
from io import BytesIO
from types import SimpleNamespace

import pytest
from mcp_types import ImageContent, TextContent
from rich.console import Console, Group
from rich.measure import Measurement
from textual_image._terminal import CellSize
from textual_image.renderable import sixel as textual_sixel_renderer

from fast_agent.command_actions.models import PluginCommandActionImage
from fast_agent.config import LoggerSettings, Settings, TerminalImageSettings
from fast_agent.mcp.prompt_render import render_content_blocks
from fast_agent.mcp.tool_result_metadata import (
    get_tool_result_media_preview,
    set_tool_result_media_preview,
)
from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.terminal_images import (
    extract_image_artifacts,
    extract_image_render_items,
    render_assistant_images,
    render_plugin_command_images_for_settings,
    render_tool_result_images,
)
from fast_agent.ui.terminal_images import halfcell as halfcell_renderer
from fast_agent.ui.terminal_images import renderer as terminal_image_renderer
from fast_agent.ui.terminal_images import sixel as sixel_renderer

_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def _image_content() -> ImageContent:
    return ImageContent(
        type="image",
        data=base64.b64encode(_PNG_BYTES).decode("ascii"),
        mime_type="image/png",
    )


def test_terminal_image_settings_accept_textual_image_sizes() -> None:
    settings = TerminalImageSettings(width="100%", height="auto")

    assert settings.width == "100%"
    assert settings.height == "auto"


def test_extract_image_artifacts_from_mcp_image_content() -> None:
    artifacts = extract_image_artifacts([_image_content()])

    assert len(artifacts) == 1
    assert artifacts[0].data == _PNG_BYTES
    assert artifacts[0].mime_type == "image/png"
    assert artifacts[0].label.startswith("[IMAGE 1: image/png,")


def test_extract_image_render_items_attaches_tool_metadata_to_last_image() -> None:
    items = extract_image_render_items(
        [
            _image_content(),
            TextContent(type="text", text="Image URL: https://example.test/image.png"),
            TextContent(type="text", text="Seed used for generation: 123"),
        ]
    )

    assert len(items) == 1
    assert items[0].metadata == (
        "Image URL: https://example.test/image.png",
        "Seed used for generation: 123",
    )


def test_tool_result_images_render_without_console_display_state() -> None:
    config = Settings(
        logger=LoggerSettings(
            terminal_images=TerminalImageSettings(
                enabled=True,
                backend="unicode",
                width="auto",
                height="auto",
            )
        )
    )

    assert render_tool_result_images(config, [_image_content()]) is not None


def test_tool_result_media_preview_is_display_only() -> None:
    from mcp_types import CallToolResult

    result = CallToolResult(
        content=[TextContent(type="text", text="Staged image for the next model call.")],
        is_error=False,
    )

    set_tool_result_media_preview(result, [_image_content()])

    assert len(result.content) == 1
    preview = get_tool_result_media_preview(result)
    assert preview is not None
    assert len(preview) == 1
    assert isinstance(preview[0], ImageContent)


def test_tool_result_image_rendering_does_not_create_console_display_state() -> None:
    display = ConsoleDisplay(
        Settings(
            logger=LoggerSettings(
                terminal_images=TerminalImageSettings(
                    enabled=True,
                    backend="unicode",
                    width="auto",
                    height="auto",
                    render_assistant=True,
                )
            )
        )
    )

    assert "_pending_tool_image_items" not in vars(display)
    assert render_tool_result_images(display.config, [_image_content()]) is not None


def test_render_assistant_images_returns_none_for_none_backend() -> None:
    config = Settings(
        logger=LoggerSettings(
            terminal_images=TerminalImageSettings(
                enabled=True,
                backend="none",
                width="auto",
                height="auto",
            )
        )
    )

    renderable = render_assistant_images(config, [_image_content()])

    assert renderable is None


def test_tool_result_images_ignore_assistant_only_switch() -> None:
    config = Settings(
        logger=LoggerSettings(
            terminal_images=TerminalImageSettings(
                enabled=True,
                backend="unicode",
                render_assistant=False,
            )
        )
    )

    assert render_assistant_images(config, [_image_content()]) is None
    assert render_tool_result_images(config, [_image_content()]) is not None


@pytest.mark.parametrize(
    "settings",
    [
        TerminalImageSettings(enabled=False, backend="unicode"),
        TerminalImageSettings(enabled=True, backend="none"),
    ],
)
def test_plugin_command_images_do_not_load_sources_when_disabled(
    monkeypatch,
    settings: TerminalImageSettings,
) -> None:
    def fail_source_load(*args, **kwargs):
        del args, kwargs
        pytest.fail("disabled plugin images should not load image sources")

    monkeypatch.setattr(terminal_image_renderer, "_artifact_from_plugin_image", fail_source_load)

    renderable = render_plugin_command_images_for_settings(
        settings,
        [PluginCommandActionImage(source="https://example.test/image.png")],
    )

    assert renderable is None


def test_textual_image_backend_missing_class_disables_rendering(monkeypatch) -> None:
    monkeypatch.delenv("HERDR_ENV", raising=False)

    class DummyImage:
        pass

    module = SimpleNamespace(Image=DummyImage)
    monkeypatch.setattr(terminal_image_renderer, "import_module", lambda name: module)

    assert terminal_image_renderer._resolve_textual_image_class("auto") is DummyImage
    assert terminal_image_renderer._resolve_textual_image_class("kitty") is None


def test_automatic_sixel_backend_uses_viewport_aware_renderer(monkeypatch) -> None:
    monkeypatch.delenv("HERDR_ENV", raising=False)

    class SixelImage:
        pass

    class ViewportAwareSixelImage:
        pass

    module = SimpleNamespace(
        Image=SixelImage,
        SixelImage=SixelImage,
    )

    def import_backend(name: str):
        if name == "fast_agent.ui.terminal_images.sixel":
            return SimpleNamespace(ViewportAwareSixelImage=ViewportAwareSixelImage)
        return module

    monkeypatch.setattr(terminal_image_renderer, "import_module", import_backend)

    assert terminal_image_renderer._resolve_textual_image_class("auto") is ViewportAwareSixelImage
    assert (
        terminal_image_renderer._resolve_textual_image_class("textual-image")
        is ViewportAwareSixelImage
    )
    assert terminal_image_renderer._resolve_textual_image_class("sixel") is ViewportAwareSixelImage


@pytest.mark.parametrize("herdr", [False, True])
def test_auto_backend_honors_reported_kitty_support(monkeypatch, herdr: bool) -> None:
    class TGPImage:
        pass

    monkeypatch.setenv("HERDR_ENV", "1" if herdr else "0")
    monkeypatch.setattr(
        terminal_image_renderer,
        "import_module",
        lambda name: SimpleNamespace(Image=TGPImage, TGPImage=TGPImage),
    )

    assert terminal_image_renderer._resolve_textual_image_class("auto") is TGPImage
    assert terminal_image_renderer._resolve_textual_image_class("kitty") is TGPImage


def test_explicit_halfcell_backend_in_herdr_has_no_fallback_warning(monkeypatch) -> None:
    monkeypatch.setenv("HERDR_ENV", "1")
    settings = TerminalImageSettings(backend="halfcell", width=1, height=1)

    renderable = terminal_image_renderer.render_image_items(
        settings,
        terminal_image_renderer.extract_image_render_items([_image_content()]),
    )

    assert isinstance(renderable, Group)
    assert len(renderable.renderables) == 2
    assert isinstance(renderable.renderables[1], halfcell_renderer.HerdrAwareHalfcellImage)


def test_herdr_halfcell_replaces_implausible_cell_geometry(monkeypatch) -> None:
    monkeypatch.setattr(halfcell_renderer, "get_cell_size", lambda: CellSize(2, 8))
    image = halfcell_renderer.HerdrAwareHalfcellImage(
        BytesIO(_PNG_BYTES),
        width="80%",
        height="auto",
    )
    console = Console(width=80, height=40, force_terminal=True)

    segments = list(console.render(image, console.options))

    assert sum(segment.text == "\n" for segment in segments) == 32
    assert Measurement.get(console, console.options, image) == Measurement(64, 64)


def test_sixel_image_height_stays_within_cursor_safe_viewport(monkeypatch) -> None:
    def cell_size() -> CellSize:
        return CellSize(10, 20)

    monkeypatch.setattr(sixel_renderer, "get_cell_size", cell_size)
    monkeypatch.setattr(textual_sixel_renderer, "get_cell_size", cell_size)
    image = sixel_renderer.ViewportAwareSixelImage(
        BytesIO(_PNG_BYTES),
        width="80%",
        height="auto",
    )
    console = Console(width=80, height=24, force_terminal=True)

    segments = list(console.render(image, console.options))
    save_cursor_index = next(
        index for index, segment in enumerate(segments) if segment.text == "\x1b7"
    )

    assert save_cursor_index == 23
    assert all(segment.text == " " * 46 + "\n" for segment in segments[:save_cursor_index])
    assert segments[save_cursor_index + 1].text == "\x1b[23A"
    assert Measurement.get(console, console.options, image) == Measurement(46, 46)

    for start_row in (1, 12, 24):
        scroll_count = max(0, start_row + save_cursor_index - console.height)
        image_row_after_scroll = start_row - scroll_count
        cursor_row_before_rewind = min(console.height, start_row + save_cursor_index)
        assert cursor_row_before_rewind - save_cursor_index == image_row_after_scroll


def test_sixel_image_is_suppressed_without_a_cursor_safe_row() -> None:
    image = sixel_renderer.ViewportAwareSixelImage(BytesIO(_PNG_BYTES))
    console = Console(width=80, height=1, force_terminal=True)

    assert list(console.render(image, console.options)) == []
    assert Measurement.get(console, console.options, image) == Measurement(0, 0)


def test_render_content_blocks_summarizes_images_without_base64_payload() -> None:
    image = _image_content()
    rendered = render_content_blocks([image])

    assert "[IMAGE: image/png," in rendered
    assert image.data not in rendered


def _png_bytes_of_size(width: int, height: int) -> bytes:
    from PIL import Image as PILImage

    buf = BytesIO()
    PILImage.new("RGBA", (width, height), (0, 0, 0, 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_aspect_ratio_fit_does_not_upscale_portrait_images(monkeypatch) -> None:
    monkeypatch.setattr(
        terminal_image_renderer,
        "_terminal_viewport_geometry",
        lambda: (180, 50, 9.0, 19.0),
    )
    settings = TerminalImageSettings(width="80%", height="auto")
    portrait = terminal_image_renderer.ImageArtifact(
        data=_png_bytes_of_size(196, 398),
        mime_type="image/png",
        label="State (test)",
    )

    cols, rows = terminal_image_renderer._resolve_render_dimensions(portrait, settings)
    assert cols == 22
    assert rows == 21


def test_aspect_ratio_fit_clamps_tall_and_wide_images_to_viewport(monkeypatch) -> None:
    monkeypatch.setattr(
        terminal_image_renderer,
        "_terminal_viewport_geometry",
        lambda: (180, 50, 9.0, 19.0),
    )
    settings = TerminalImageSettings(width="80%", height="auto")

    tall = terminal_image_renderer.ImageArtifact(
        data=_png_bytes_of_size(400, 1600),
        mime_type="image/png",
        label="Tall",
    )
    tall_cols, tall_rows = terminal_image_renderer._resolve_render_dimensions(tall, settings)
    assert tall_rows == 30  # 60% of 50 rows
    assert tall_cols == 16

    wide = terminal_image_renderer.ImageArtifact(
        data=_png_bytes_of_size(1800, 400),
        mime_type="image/png",
        label="Wide",
    )
    wide_cols, wide_rows = terminal_image_renderer._resolve_render_dimensions(wide, settings)
    assert wide_cols == 144  # 80% of 180 cols
    assert wide_rows == 15


def test_plugin_image_width_and_height_hints_are_honored_and_clamped(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        terminal_image_renderer,
        "_terminal_viewport_geometry",
        lambda: (180, 50, 9.0, 19.0),
    )
    img_path = tmp_path / "wide.png"
    img_path.write_bytes(_png_bytes_of_size(1800, 400))

    captured: list[tuple[int | str | None, int | str | None]] = []

    class FakeImage:
        def __init__(self, _stream, *, width=None, height=None) -> None:
            captured.append((width, height))

    monkeypatch.setattr(
        terminal_image_renderer,
        "_resolve_textual_image_class",
        lambda _backend: FakeImage,
    )

    settings = TerminalImageSettings(enabled=True, backend="kitty", width="80%", height="auto")
    # Plugin suggests a narrower width (50 cells) -> honored
    render_plugin_command_images_for_settings(
        settings,
        [PluginCommandActionImage(source=str(img_path), width=50)],
    )
    assert captured[-1] == (50, 5)

    # Plugin suggests 100% width -> clamped to user settings (80% = 144 cells)
    render_plugin_command_images_for_settings(
        settings,
        [PluginCommandActionImage(source=str(img_path), width="100%")],
    )
    assert captured[-1] == (144, 15)
