from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from urllib.parse import SplitResult, urlsplit, urlunsplit

from rich.text import Text

from fast_agent.constants import ANTHROPIC_CITATIONS_CHANNEL, ANTHROPIC_SERVER_TOOLS_CHANNEL
from fast_agent.utils.markdown import escape_markdown_text
from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from fast_agent.types import PromptMessageExtended


type JsonObject = dict[str, object]


@runtime_checkable
class _TextPayloadBlock(Protocol):
    text: str


@dataclass(frozen=True)
class CitationSource:
    index: int
    title: str | None
    url: str | None
    source: str | None

    @property
    def display_title(self) -> str:
        return self.title or self.source or f"Source {self.index}"


_WEB_TOOL_BADGE_ORDER = ("web_search", "web_fetch")


def _iter_channel_payloads(
    channels: Mapping[str, Sequence[object]] | None,
    channel_name: str,
) -> list[JsonObject]:
    if not channels:
        return []

    channel_blocks = channels.get(channel_name)
    if not channel_blocks:
        return []

    payloads: list[JsonObject] = []
    for block in channel_blocks:
        if not isinstance(block, _TextPayloadBlock):
            continue
        text = block.text
        if not isinstance(text, str) or not text:
            continue
        try:
            decoded = json.loads(text)
        except JSONDecodeError:
            continue

        payload = _json_object(decoded)
        if payload is not None:
            payloads.append(payload)
        elif isinstance(decoded, list):
            payloads.extend(
                item_payload for item in decoded if (item_payload := _json_object(item)) is not None
            )

    return payloads


def _json_object(value: object) -> JsonObject | None:
    if not isinstance(value, dict):
        return None
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _string_field(payload: JsonObject, key: str) -> str | None:
    value = payload.get(key)
    return value if isinstance(value, str) else None


def _normalize_url(url: str | None) -> str | None:
    if not url:
        return None
    normalized = url.strip()
    if not normalized:
        return None

    try:
        split = urlsplit(normalized)
    except ValueError:
        return normalized

    if not split.scheme or not split.netloc:
        return normalized

    scheme = strip_casefold(split.scheme)
    netloc = _normalize_netloc(split, scheme)
    path = split.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    return urlunsplit((scheme, netloc, path, split.query, ""))


def _normalize_netloc(split: SplitResult, scheme: str) -> str:
    netloc = strip_casefold(split.netloc)
    default_port = {"http": 80, "https": 443}.get(scheme)
    if default_port is None:
        return netloc
    try:
        port = split.port
    except ValueError:
        return netloc
    if port == default_port:
        return netloc.rsplit(f":{default_port}", 1)[0]
    return netloc


def _escape_markdown_link_destination(url: str) -> str:
    return url.replace("\\", "\\\\").replace("(", r"\(").replace(")", r"\)")


def _metadata_key_part(value: str | None) -> str:
    return strip_casefold(value) if value is not None else ""


def _citation_source_key(
    *,
    title: str | None,
    source: str | None,
    normalized_url: str | None,
) -> tuple[str, str] | tuple[str, str, str] | None:
    if normalized_url:
        return ("url", normalized_url)

    title_key = _metadata_key_part(title)
    source_key = _metadata_key_part(source)
    if not title_key and not source_key:
        return None
    return ("meta", title_key, source_key)


def collect_citation_sources(message: "PromptMessageExtended") -> list[CitationSource]:
    payloads = _iter_channel_payloads(message.channels, ANTHROPIC_CITATIONS_CHANNEL)
    if not payloads:
        return []

    seen: set[tuple[str, str] | tuple[str, str, str]] = set()
    sources: list[CitationSource] = []

    for payload in payloads:
        title = _string_field(payload, "title")
        source = _string_field(payload, "source")

        raw_url = _string_field(payload, "url")
        normalized_url = _normalize_url(raw_url)

        key = _citation_source_key(
            title=title,
            source=source,
            normalized_url=normalized_url,
        )
        if key is None:
            continue
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
        display_title = escape_markdown_text(source.display_title)
        if source.url:
            destination = _escape_markdown_link_destination(source.url)
            lines.append(f"- [{source.index}] [{display_title}]({destination})")
        else:
            lines.append(f"- [{source.index}] {display_title}")
    return "\n".join(lines)


def _render_sources_text(
    message: "PromptMessageExtended",
    *,
    leading_breaks: int,
) -> Text | None:
    sources = collect_citation_sources(message)
    if not sources:
        return None

    rendered = Text("\n" * leading_breaks + "Sources\n")
    for source in sources:
        rendered.append(" ")
        rendered.append(f"[{source.index}]", style="bright_green")
        rendered.append(f" {source.display_title}", style="bright_white")
        if source.url:
            rendered.append(" — ", style="bright_white")
            rendered.append(source.url, style="bright_blue underline")
        rendered.append("\n")

    return rendered


def render_sources_additional_text(message: "PromptMessageExtended") -> Text | None:
    """Render citations as styled multi-line text for post-answer console output."""
    return _render_sources_text(message, leading_breaks=2)


def render_sources_pre_content(message: "PromptMessageExtended") -> Text | None:
    """Render citations as styled multi-line text for pre-answer console output."""
    return _render_sources_text(message, leading_breaks=0)


def web_tool_badges(message: "PromptMessageExtended") -> list[str]:
    payloads = _iter_channel_payloads(message.channels, ANTHROPIC_SERVER_TOOLS_CHANNEL)
    if not payloads:
        return []

    counts = dict.fromkeys(_WEB_TOOL_BADGE_ORDER, 0)
    for payload in payloads:
        tool_type = _string_field(payload, "type")
        if tool_type == "server_tool_use":
            name = _string_field(payload, "name")
            if name in counts:
                counts[name] += 1

    return [f"{name} x{counts[name]}" for name in _WEB_TOOL_BADGE_ORDER if counts[name]]
