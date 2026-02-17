from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from rich.text import Text

from fast_agent.constants import ANTHROPIC_CITATIONS_CHANNEL, ANTHROPIC_SERVER_TOOLS_CHANNEL

if TYPE_CHECKING:
    from fast_agent.types import PromptMessageExtended


@dataclass(frozen=True)
class CitationSource:
    index: int
    title: str | None
    url: str | None
    source: str | None


def _iter_channel_payloads(
    channels: Mapping[str, Sequence[object]] | None,
    channel_name: str,
) -> list[dict[str, Any]]:
    if not channels:
        return []

    channel_blocks = channels.get(channel_name)
    if not channel_blocks:
        return []

    payloads: list[dict[str, Any]] = []
    for block in channel_blocks:
        text = getattr(block, "text", None)
        if not isinstance(text, str) or not text:
            continue
        try:
            decoded = json.loads(text)
        except Exception:
            continue

        if isinstance(decoded, dict):
            payloads.append(decoded)
        elif isinstance(decoded, list):
            payloads.extend(item for item in decoded if isinstance(item, dict))

    return payloads


def _normalize_url(url: str | None) -> str | None:
    if not url:
        return None
    normalized = url.strip()
    if not normalized:
        return None

    try:
        split = urlsplit(normalized)
    except Exception:
        return normalized

    if not split.scheme or not split.netloc:
        return normalized

    scheme = split.scheme.lower()
    netloc = split.netloc.lower()
    path = split.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    return urlunsplit((scheme, netloc, path, split.query, ""))


def collect_citation_sources(message: "PromptMessageExtended") -> list[CitationSource]:
    payloads = _iter_channel_payloads(message.channels, ANTHROPIC_CITATIONS_CHANNEL)
    if not payloads:
        return []

    seen: set[tuple[str, str] | tuple[str, str, str]] = set()
    sources: list[CitationSource] = []

    for payload in payloads:
        title = payload.get("title") if isinstance(payload.get("title"), str) else None
        source = payload.get("source") if isinstance(payload.get("source"), str) else None

        raw_url = payload.get("url") if isinstance(payload.get("url"), str) else None
        normalized_url = _normalize_url(raw_url)

        if normalized_url:
            key: tuple[str, str] | tuple[str, str, str] = ("url", normalized_url)
        else:
            title_key = (title or "").strip().lower()
            source_key = (source or "").strip().lower()
            if not title_key and not source_key:
                continue
            key = ("meta", title_key, source_key)

        if key in seen:
            continue
        seen.add(key)
        sources.append(
            CitationSource(
                index=len(sources) + 1,
                title=title,
                url=normalized_url,
                source=source,
            )
        )

    return sources


def render_sources_footer(message: "PromptMessageExtended") -> str | None:
    sources = collect_citation_sources(message)
    if not sources:
        return None

    lines: list[str] = ["", "Sources", ""]
    for source in sources:
        title = source.title or source.source or f"Source {source.index}"
        if source.url:
            lines.append(f"- [{source.index}] [{title}]({source.url})")
        else:
            lines.append(f"- [{source.index}] {title}")
    return "\n".join(lines)


def render_sources_additional_text(message: "PromptMessageExtended") -> Text | None:
    """Render citations as styled multi-line text for console output."""
    sources = collect_citation_sources(message)
    if not sources:
        return None

    rendered = Text("\n\nSources\n")
    for source in sources:
        title = source.title or source.source or f"Source {source.index}"
        rendered.append(" ")
        rendered.append(f"[{source.index}]", style="bright_green")
        rendered.append(f" {title}", style="bright_white")
        if source.url:
            rendered.append(" — ", style="bright_white")
            rendered.append(source.url, style="bright_blue underline")
        rendered.append("\n")

    return rendered


def web_tool_badges(message: "PromptMessageExtended") -> list[str]:
    payloads = _iter_channel_payloads(message.channels, ANTHROPIC_SERVER_TOOLS_CHANNEL)
    if not payloads:
        return []

    counts: dict[str, int] = {"web_search": 0, "web_fetch": 0}
    for payload in payloads:
        tool_type = payload.get("type")
        if tool_type == "server_tool_use":
            name = payload.get("name")
            if name in counts:
                counts[name] += 1

    badges: list[str] = []
    if counts["web_search"]:
        badges.append(f"web_search x{counts['web_search']}")
    if counts["web_fetch"]:
        badges.append(f"web_fetch x{counts['web_fetch']}")
    return badges
