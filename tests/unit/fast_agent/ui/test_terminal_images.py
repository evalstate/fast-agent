import base64
from types import SimpleNamespace

import pytest
from mcp.types import ImageContent, TextContent

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
from fast_agent.ui.terminal_images import renderer as terminal_image_renderer

_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)


def _image_content() -> ImageContent:
    return ImageContent(
        type="image",
        data=base64.b64encode(_PNG_BYTES).decode("ascii"),
        mimeType="image/png",
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
    from mcp.types import CallToolResult

    result = CallToolResult(
        content=[TextContent(type="text", text="Staged image for the next model call.")],
        isError=False,
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
    class DummyImage:
        pass

    module = SimpleNamespace(Image=DummyImage)
    monkeypatch.setattr(terminal_image_renderer, "import_module", lambda name: module)

    assert terminal_image_renderer._resolve_textual_image_class("auto") is DummyImage
    assert terminal_image_renderer._resolve_textual_image_class("kitty") is None


def test_render_content_blocks_summarizes_images_without_base64_payload() -> None:
    image = _image_content()
    rendered = render_content_blocks([image])

    assert "[IMAGE: image/png," in rendered
    assert image.data not in rendered
