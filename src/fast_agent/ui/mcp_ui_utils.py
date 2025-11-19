from __future__ import annotations

import base64
import platform
import re
import subprocess
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from mcp.types import BlobResourceContents, EmbeddedResource, TextResourceContents

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


@dataclass
class UILink:
    title: str
    file_path: str  # absolute path to local html file
    web_url: Optional[str] = None  # Preferable clickable link (http(s) or data URL)


def _safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    return name[:120] if len(name) > 120 else name


def _ensure_output_dir() -> Path:
    # Read output directory from settings, defaulting to .fast-agent/ui
    try:
        from fast_agent.config import get_settings

        settings = get_settings()
        dir_setting = getattr(settings, "mcp_ui_output_dir", ".fast-agent/ui") or ".fast-agent/ui"
    except Exception:
        dir_setting = ".fast-agent/ui"

    base = Path(dir_setting)
    if not base.is_absolute():
        base = Path.cwd() / base
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


def _decode_text_or_blob(resource) -> Optional[str]:
    """Return string content from TextResourceContents or BlobResourceContents."""
    if isinstance(resource, TextResourceContents):
        return resource.text or ""
    if isinstance(resource, BlobResourceContents):
        try:
            return base64.b64decode(resource.blob or "").decode("utf-8", errors="replace")
        except Exception:
            return None
    return None


def _first_https_url_from_uri_list(text: str) -> Optional[str]:
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("http://") or line.startswith("https://"):
            return line
    return None


def _make_html_for_raw_html(html_string: str) -> str:
    # Wrap with minimal HTML and sandbox guidance (iframe srcdoc will be used by browsers)
    return html_string


def _make_html_for_uri(url: str) -> str:
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
    <iframe src=\"{url}\" sandbox=\"allow-scripts allow-forms allow-same-origin\" referrerpolicy=\"no-referrer\"></iframe>
  </body>
  </html>
"""


def _write_html_file(name_hint: str, html: str) -> str:
    out_dir = _ensure_output_dir()
    file_name = _safe_filename(name_hint or "ui") + ".html"
    out_path = out_dir / file_name
    # Ensure unique filename if exists
    i = 1
    while out_path.exists():
        out_path = out_dir / f"{_safe_filename(name_hint)}_{i}.html"
        i += 1
    out_path.write_text(html, encoding="utf-8")
    return str(out_path.resolve())


def ui_links_from_channel(resources: Iterable[EmbeddedResource]) -> List[UILink]:
    """
    Build local HTML files for a list of MCP-UI EmbeddedResources and return clickable links.

    Supported mime types:
    - text/html: expects text or base64 blob of HTML
    - text/uri-list: expects text or blob of a single URL (first valid URL is used)
    - application/vnd.mcp-ui.remote-dom* : currently unsupported; generate a placeholder page
    """
    links: List[UILink] = []
    for emb in resources:
        res = emb.resource
        uri = str(getattr(res, "uri", "")) if getattr(res, "uri", None) else None
        mime = getattr(res, "mimeType", "") or ""
        title = _extract_title(uri)
        content = _decode_text_or_blob(res)

        if mime.startswith("text/html"):
            if content is None:
                continue
            html = _make_html_for_raw_html(content)
            file_path = _write_html_file(title, html)
            # Generate data URL only if enabled
            if ENABLE_DATA_URLS:
                try:
                    b64 = base64.b64encode(html.encode("utf-8")).decode("ascii")
                    data_url = f"data:text/html;base64,{b64}"
                    # Some terminals have limits; only attach when reasonably small
                    web_url = data_url if len(data_url) < 12000 else None
                except Exception:
                    web_url = None
            else:
                web_url = None
            links.append(UILink(title=title, file_path=file_path, web_url=web_url))

        elif mime.startswith("text/uri-list"):
            if content is None:
                continue
            url = _first_https_url_from_uri_list(content)
            if not url:
                # fallback: try to treat entire content as a URL
                url = content.strip()
            if not (url and (url.startswith("http://") or url.startswith("https://"))):
                continue
            html = _make_html_for_uri(url)
            file_path = _write_html_file(title, html)
            # Prefer the direct URL for clickability; keep file for archival
            links.append(UILink(title=title, file_path=file_path, web_url=url))

        elif mime.startswith("application/vnd.mcp-ui.remote-dom"):
            # Not supported yet - generate informational page
            placeholder = f"""
<!doctype html>
<html><head><meta charset=\"utf-8\" /><title>{title} (Unsupported)</title></head>
<body>
  <p>Remote DOM resources are not supported yet in this client.</p>
  <p>URI: {uri or ""}</p>
  <p>mimeType: {mime}</p>
  <pre style=\"white-space: pre-wrap;\">{(content or "")[:4000]}</pre>
  <p>Please upgrade fast-agent when support becomes available.</p>
  </body></html>
"""
            file_path = _write_html_file(title + "_unsupported", placeholder)
            links.append(UILink(title=title + " (unsupported)", file_path=file_path))
        else:
            # Unknown, skip quietly
            continue

    return links


def open_links_in_browser(links: Iterable[UILink], mcp_ui_mode: str = "auto") -> None:
    """Open links in browser/system viewer.

    Args:
        links: Links to open
        mcp_ui_mode: UI mode setting ("disabled", "enabled", "auto")
    """
    # Only attempt to open files when in auto mode
    if mcp_ui_mode != "auto":
        return

    for link in links:
        try:
            # Use subprocess for better file:// handling across platforms
            file_path = link.file_path

            system = platform.system()
            if system == "Darwin":  # macOS
                subprocess.run(["open", file_path], check=False, capture_output=True)
            elif system == "Windows":
                subprocess.run(
                    ["start", "", file_path], shell=True, check=False, capture_output=True
                )
            elif system == "Linux":
                # Try xdg-open first (most common), fallback to other options
                try:
                    subprocess.run(["xdg-open", file_path], check=False, capture_output=True)
                except FileNotFoundError:
                    # Fallback to webbrowser for Linux if xdg-open not available
                    webbrowser.open(f"file://{file_path}", new=2)
            else:
                # Unknown system, fallback to webbrowser
                webbrowser.open(f"file://{file_path}", new=2)
        except Exception:
            # Silently ignore errors - user can still manually open the file
            pass
