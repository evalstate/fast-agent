from __future__ import annotations

from pathlib import Path

from mcp.types import EmbeddedResource, TextResourceContents
from pydantic import AnyUrl

from fast_agent.ui import mcp_ui_utils


def test_write_html_file_reuses_default_base_name_for_collisions(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(mcp_ui_utils, "_ensure_output_dir", lambda: tmp_path)
    (tmp_path / "ui.html").write_text("existing", encoding="utf-8")

    path = Path(mcp_ui_utils._write_html_file("", "<p>new</p>"))

    assert path.name == "ui_1.html"
    assert path.read_text(encoding="utf-8") == "<p>new</p>"


def test_write_html_file_falls_back_when_sanitized_name_is_empty(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(mcp_ui_utils, "_ensure_output_dir", lambda: tmp_path)

    path = Path(mcp_ui_utils._write_html_file("", "<p>ui</p>"))

    assert path.name == "ui.html"


def test_write_html_file_falls_back_for_separator_only_name(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(mcp_ui_utils, "_ensure_output_dir", lambda: tmp_path)

    path = Path(mcp_ui_utils._write_html_file("...", "<p>ui</p>"))

    assert path.name == "ui.html"


def test_make_html_for_uri_escapes_iframe_src_attribute() -> None:
    html = mcp_ui_utils._make_html_for_uri('https://example.test/?q="><script>')

    assert 'src="https://example.test/?q=&quot;&gt;&lt;script&gt;"' in html
    assert 'src="https://example.test/?q="><script>"' not in html


def test_open_file_in_system_viewer_uses_platform_command(monkeypatch) -> None:
    calls: list[tuple[list[str], bool, bool]] = []
    monkeypatch.setattr(mcp_ui_utils.platform, "system", lambda: "Darwin")

    def record_run(command: list[str], *, check: bool, capture_output: bool):
        calls.append((command, check, capture_output))

    monkeypatch.setattr(mcp_ui_utils.subprocess, "run", record_run)

    mcp_ui_utils._open_file_in_system_viewer("/tmp/ui.html")

    assert calls == [(["open", "/tmp/ui.html"], False, True)]


def test_open_file_in_system_viewer_falls_back_when_command_missing(monkeypatch) -> None:
    opened: list[tuple[str, int]] = []
    monkeypatch.setattr(mcp_ui_utils.platform, "system", lambda: "Linux")

    def missing_command(command: list[str], *, check: bool, capture_output: bool):
        raise FileNotFoundError(command[0])

    monkeypatch.setattr(mcp_ui_utils.subprocess, "run", missing_command)
    monkeypatch.setattr(
        mcp_ui_utils.webbrowser,
        "open",
        lambda url, new=0: opened.append((url, new)),
    )

    mcp_ui_utils._open_file_in_system_viewer("/tmp/ui.html")

    assert opened == [("file:///tmp/ui.html", 2)]


def test_remote_dom_placeholder_escapes_resource_fields(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(mcp_ui_utils, "_ensure_output_dir", lambda: tmp_path)
    resource = EmbeddedResource(
        type="resource",
        resource=TextResourceContents(
            uri=AnyUrl("ui://card/<unsafe>"),
            mimeType="application/vnd.mcp-ui.remote-dom",
            text="<script>alert(1)</script>",
        ),
    )

    links = mcp_ui_utils.ui_links_from_channel([resource])

    assert len(links) == 1
    html = Path(links[0].file_path).read_text(encoding="utf-8")
    assert "%3Cunsafe%3E" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
