from __future__ import annotations

import base64
import html as html_lib
import os
import platform
import re
import subprocess
import webbrowser
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

from mcp.types import BlobResourceContents, ContentBlock, EmbeddedResource, TextResourceContents

from fast_agent.paths import resolve_mcp_ui_output_dir
from fast_agent.utils.filename import sanitize_filename_component

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from fast_agent.mcp.ui_modes import McpUIMode


"""
Utilities for handling MCP-UI resources carried in PromptMessageExtended.channels.

Responsibilities:
- Identify MCP-UI EmbeddedResources from channels
- Decode text/blob content depending on mimeType
- Produce local HTML files that safely embed the UI content (srcdoc or iframe)
- Return presentable link labels for console display
"""

# Control whether to generate data URLs for embedded HTML content
# When disabled, always use file:// URLs which work better with most terminals
ENABLE_DATA_URLS = False
_OPEN_FILE_COMMANDS = {
    "Darwin": ("open",),
    "Linux": ("xdg-open",),
}


@dataclass
class UILink:
    title: str
    file_path: str  # absolute path to local html file
    web_url: str | None = None  # Preferable clickable link (http(s) or data URL)


type UIResourceContents = TextResourceContents | BlobResourceContents


def _ensure_output_dir() -> Path:
    base = resolve_mcp_ui_output_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base


def _extract_title(uri: str | None) -> str:
    if not uri:
        return "UI"
    try:
        # ui://component/instance -> component:instance
        without_scheme = uri.split("ui://", 1)[1] if uri.startswith("ui://") else uri
        parts = [p for p in re.split(r"[/:]", without_scheme) if p]
        if len(parts) >= 2:
            return f"{parts[0]}:{parts[1]}"
        return parts[0] if parts else "UI"
    except Exception:
        return "UI"


def _decode_text_or_blob(resource: UIResourceContents) -> str | None:
    """Return string content from TextResourceContents or BlobResourceContents."""
    if isinstance(resource, TextResourceContents):
        return resource.text or ""
    if isinstance(resource, BlobResourceContents):
        try:
            return base64.b64decode(resource.blob or "").decode("utf-8", errors="replace")
        except Exception:
            return None
    return None


def _resource_uri(resource: UIResourceContents) -> str | None:
    return str(resource.uri) if resource.uri else None


def _resource_mime_type(resource: UIResourceContents) -> str:
    return resource.mimeType or ""


def _first_https_url_from_uri_list(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(("http://", "https://")):
            return stripped
    return None


def _make_html_for_raw_html(html_string: str) -> str:
    # Wrap with minimal HTML and sandbox guidance (iframe srcdoc will be used by browsers)
    return html_string


def _make_html_for_uri(url: str) -> str:
    escaped_url = html_lib.escape(url, quote=True)
    return f"""
<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>MCP-UI</title>
    <style>html,body,iframe{{margin:0;padding:0;height:100%;width:100%;border:0}}</style>
  </head>
  <body>
    <iframe src=\"{escaped_url}\" sandbox=\"allow-scripts allow-forms allow-same-origin\" referrerpolicy=\"no-referrer\"></iframe>
  </body>
  </html>
"""


def _write_html_file(name_hint: str, html: str) -> str:
    out_dir = _ensure_output_dir()
    base_name = sanitize_filename_component(name_hint or "ui", fallback="ui")[:120]
    file_name = base_name + ".html"
    out_path = out_dir / file_name
    # Ensure unique filename if exists
    i = 1
    while out_path.exists():
        out_path = out_dir / f"{base_name}_{i}.html"
        i += 1
    out_path.write_text(html, encoding="utf-8")
    return str(out_path.resolve())


def _data_url_for_html(html: str) -> str | None:
    if not ENABLE_DATA_URLS:
        return None
    try:
        b64 = base64.b64encode(html.encode("utf-8")).decode("ascii")
    except Exception:
        return None
    data_url = f"data:text/html;base64,{b64}"
    return data_url if len(data_url) < 12000 else None


def _html_ui_link(title: str, content: str) -> UILink:
    html = _make_html_for_raw_html(content)
    file_path = _write_html_file(title, html)
    return UILink(
        title=title,
        file_path=file_path,
        web_url=_data_url_for_html(html),
    )


def _uri_list_ui_link(title: str, content: str) -> UILink | None:
    url = _first_https_url_from_uri_list(content) or content.strip()
    if not url.startswith(("http://", "https://")):
        return None
    html = _make_html_for_uri(url)
    file_path = _write_html_file(title, html)
    return UILink(title=title, file_path=file_path, web_url=url)


def _remote_dom_placeholder_link(
    title: str,
    uri: str | None,
    mime: str,
    content: str | None,
) -> UILink:
    escaped_title = html_lib.escape(title, quote=False)
    escaped_uri = html_lib.escape(uri or "", quote=False)
    escaped_mime = html_lib.escape(mime, quote=False)
    escaped_content = html_lib.escape((content or "")[:4000], quote=False)
    placeholder = f"""
<!doctype html>
<html><head><meta charset=\"utf-8\" /><title>{escaped_title} (Unsupported)</title></head>
<body>
  <p>Remote DOM resources are not supported yet in this client.</p>
  <p>URI: {escaped_uri}</p>
  <p>mimeType: {escaped_mime}</p>
  <pre style=\"white-space: pre-wrap;\">{escaped_content}</pre>
  <p>Please upgrade fast-agent when support becomes available.</p>
  </body></html>
"""
    file_path = _write_html_file(title + "_unsupported", placeholder)
    return UILink(title=title + " (unsupported)", file_path=file_path)


def _ui_link_from_resource(resource: UIResourceContents) -> UILink | None:
    uri = _resource_uri(resource)
    mime = _resource_mime_type(resource)
    title = _extract_title(uri)
    content = _decode_text_or_blob(resource)

    if mime.startswith("text/html"):
        return _html_ui_link(title, content) if content is not None else None
    if mime.startswith("text/uri-list"):
        return _uri_list_ui_link(title, content) if content is not None else None
    if mime.startswith("application/vnd.mcp-ui.remote-dom"):
        return _remote_dom_placeholder_link(title, uri, mime, content)
    return None


def ui_links_from_channel(resources: Iterable[ContentBlock]) -> list[UILink]:
    """
    Build local HTML files for a list of MCP-UI EmbeddedResources and return clickable links.

    Supported mime types:
    - text/html: expects text or base64 blob of HTML
    - text/uri-list: expects text or blob of a single URL (first valid URL is used)
    - application/vnd.mcp-ui.remote-dom* : currently unsupported; generate a placeholder page
    """
    links: list[UILink] = []
    for item in resources:
        if not isinstance(item, EmbeddedResource):
            continue
        link = _ui_link_from_resource(item.resource)
        if link is not None:
            links.append(link)

    return links


def _open_file_in_system_viewer(file_path: str) -> None:
    system = platform.system()
    if system == "Windows":
        os.startfile(file_path)  # ty: ignore[unresolved-attribute]
        return

    command = _OPEN_FILE_COMMANDS.get(system)
    if command is None:
        webbrowser.open(f"file://{file_path}", new=2)
        return

    try:
        subprocess.run([*command, file_path], check=False, capture_output=True)
    except FileNotFoundError:
        webbrowser.open(f"file://{file_path}", new=2)


def open_links_in_browser(links: Iterable[UILink], mcp_ui_mode: "McpUIMode" = "auto") -> None:
    """Open links in browser/system viewer.

    Args:
        links: Links to open
        mcp_ui_mode: UI mode setting ("disabled", "enabled", "auto")
    """
    # Only attempt to open files when in auto mode
    if mcp_ui_mode != "auto":
        return

    for link in links:
        with suppress(Exception):
            _open_file_in_system_viewer(link.file_path)
