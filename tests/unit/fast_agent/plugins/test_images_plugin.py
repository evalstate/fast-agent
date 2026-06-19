"""Unit coverage for the bundled `images` command plugin extractor.

Loaded directly from the plugin source file (the plugin ships outside the
``fast_agent`` package, by path), so this mirrors the runtime loader rather
than importing a packaged module path.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.types import TextContent

_PLUGIN_SOURCE = (
    Path(__file__).resolve().parents[4] / "plugins" / "images" / "images.py"
)


@pytest.fixture(scope="module")
def plugin():
    spec = importlib.util.spec_from_file_location(
        "fast_agent_tests.images_plugin",
        _PLUGIN_SOURCE,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec_module: dataclass processing looks up the module.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def test_detects_real_image_paths_and_urls(plugin) -> None:
    for source in (
        "/tmp/output.png",
        "./photo.png",
        "output.png",
        "render-42.webp",
        "~/pics/a.jpg",
        "https://example.com/a.png",
        "https://example.com/path/to/c.jpeg",
    ):
        assert plugin._looks_like_image_source(source), source


def test_rejects_bare_suffix_tokens(plugin) -> None:
    for suffix in plugin._IMAGE_SUFFIXES:
        assert plugin._looks_like_image_source(suffix) is False


def test_rejects_slash_joined_suffix_listing(plugin) -> None:
    listing = "/".join(plugin._IMAGE_SUFFIXES)
    assert plugin._looks_like_image_source(listing) is False


def test_rejects_non_image_text(plugin) -> None:
    for source in ("not-an-image.txt", "article.md", "https://example.com/"):
        assert plugin._looks_like_image_source(source) is False


def test_still_accepts_gradio_file_payloads(plugin) -> None:
    assert plugin._looks_like_image_source(
        "http://host/gradio_api/file=/tmp/x/y.tiff"
    )


def test_scan_filters_false_positives_keeps_real_path(plugin) -> None:
    text = (
        "Supported formats: .png .jpg .jpeg .webp .gif .bmp .tif .tiff\n"
        "Output saved to: /tmp/output.png"
    )
    candidates = plugin._extract_from_text(text, "tool result 5")
    assert [c.source for c in candidates] == ["/tmp/output.png"]


def test_scan_filters_combined_suffix_listing_in_assistant_text(plugin) -> None:
    text = "fix these: .png/.jpg/.jpeg/.webp/.gif/.bmp/.tif/.tiff now"
    assert plugin._extract_from_text(text, "assistant message 6") == []


def _msg(role: str, text: str) -> SimpleNamespace:
    return SimpleNamespace(role=role, content=[TextContent(type="text", text=text)])


def _history() -> list[SimpleNamespace]:
    # An older assistant image, then a user turn, then a fresh assistant image.
    return [
        _msg("assistant", "old turn: /tmp/old.png"),
        _msg("user", "show me something"),
        _msg("assistant", "here it is: /tmp/new.png"),
    ]


def _ctx(arguments: str) -> SimpleNamespace:
    return SimpleNamespace(arguments=arguments, message_history=_history())


def test_bare_images_lists_only_last_turn(plugin) -> None:
    result = asyncio.run(plugin.images(_ctx("")))

    assert result.markdown is not None
    assert "/tmp/new.png" in result.markdown
    assert "/tmp/old.png" not in result.markdown
    assert "last user turn" in result.markdown


def test_images_list_shows_everything(plugin) -> None:
    result = asyncio.run(plugin.images(_ctx("list")))

    assert result.markdown is not None
    assert "/tmp/old.png" in result.markdown
    assert "/tmp/new.png" in result.markdown
    assert "Recent image sources" in result.markdown


def test_images_explicit_last_matches_bare(plugin) -> None:
    explicit = asyncio.run(plugin.images(_ctx("last")))
    bare = asyncio.run(plugin.images(_ctx("")))

    assert explicit.markdown == bare.markdown