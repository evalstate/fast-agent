import base64

from mcp.types import ImageContent, TextContent

from fast_agent.config import LoggerSettings, Settings, TerminalImageSettings
from fast_agent.mcp.prompt_render import render_content_blocks
from fast_agent.ui.console_display import ConsoleDisplay
from fast_agent.ui.terminal_images import (
    extract_image_artifacts,
    extract_image_render_items,
    render_terminal_images,
)

_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ"
    "/pLvAAAAAElFTkSuQmCC"
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


def test_console_display_drains_tool_images_into_final_assistant_pass() -> None:
    display = ConsoleDisplay(
        Settings(
            logger=LoggerSettings(
                terminal_images=TerminalImageSettings(
                    enabled=True,
                    backend="unicode",
                    width="auto",
                    height="auto",
                )
            )
        )
    )
    display.collect_tool_result_images([_image_content()])

    assert display._drain_tool_result_images() is not None
    assert display._drain_tool_result_images() is None


def test_tool_image_rendering_setting_still_uses_final_assistant_pass() -> None:
    display = ConsoleDisplay(
        Settings(
            logger=LoggerSettings(
                terminal_images=TerminalImageSettings(
                    enabled=True,
                    backend="unicode",
                    width="auto",
                    height="auto",
                    render_tools=True,
                    render_assistant=True,
                )
            )
        )
    )
    display.collect_tool_result_images([_image_content()])

    assert render_terminal_images(display.config, "tools", [_image_content()]) is None
    assert display._drain_tool_result_images() is not None


def test_console_display_can_render_pending_tool_images_without_assistant_reprint() -> None:
    display = ConsoleDisplay(
        Settings(
            logger=LoggerSettings(
                terminal_images=TerminalImageSettings(
                    enabled=True,
                    backend="unicode",
                    width="auto",
                    height="auto",
                )
            )
        )
    )
    display.collect_tool_result_images([_image_content()])

    display.show_pending_tool_result_images()

    assert display._drain_tool_result_images() is None


def test_render_terminal_images_returns_none_for_none_backend() -> None:
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

    renderable = render_terminal_images(config, "tools", [_image_content()])

    assert renderable is None


def test_render_content_blocks_summarizes_images_without_base64_payload() -> None:
    image = _image_content()
    rendered = render_content_blocks([image])

    assert "[IMAGE: image/png," in rendered
    assert image.data not in rendered
