from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Iterable

from fast_agent.command_actions import (
    PluginCommandActionImage,
    PluginCommandActionResult,
    PluginCommandCompletion,
)
from fast_agent.mcp.helpers.content_helpers import get_text

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[(?P<label>[^\]]*)\]\((?P<source>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
_MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[(?P<label>[^\]]+)\]\((?P<source>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
_URL_RE = re.compile(r"https?://[^\s<>)\]]+")
_FILE_URL_RE = re.compile(r"file://[^\s<>)\]]+")
_PATH_RE = re.compile(r"(?<![\w:/.-])(?:~|\.|\.\.|/)[^\s<>)\]]+", re.MULTILINE)
_TRAILING_PUNCT = ".,;:'\"`]}>)"


@dataclass(frozen=True)
class ImageCandidate:
    source: str
    label: str | None
    origin: str


def _clean_source(source: str) -> str:
    return source.strip().rstrip(_TRAILING_PUNCT)


def _looks_like_image_source(source: str) -> bool:
    clean = _clean_source(source).lower()
    for suffix in _IMAGE_SUFFIXES:
        if not clean.endswith(suffix):
            continue
        # Require a real filename stem before the suffix, so bare extensions
        # (".png") and slash-joined lists of them (".png/.jpg/...") are not
        # treated as image paths.
        if clean.rsplit("/", 1)[-1][: -len(suffix)]:
            return True
    return "/gradio_api/file=" in clean and any(suffix in clean for suffix in _IMAGE_SUFFIXES)


def _candidate(source: str, label: str | None, origin: str) -> ImageCandidate | None:
    clean = _clean_source(source)
    if not clean or not _looks_like_image_source(clean):
        return None
    return ImageCandidate(source=clean, label=(label or None), origin=origin)


def _extract_from_text(text: str, origin: str) -> list[ImageCandidate]:
    candidates: list[ImageCandidate] = []
    occupied: list[tuple[int, int]] = []

    for pattern in (_MARKDOWN_IMAGE_RE, _MARKDOWN_LINK_RE):
        for match in pattern.finditer(text):
            if item := _candidate(match.group("source"), match.group("label"), origin):
                candidates.append(item)
                occupied.append(match.span("source"))

    def protected(start: int) -> bool:
        return any(left <= start < right for left, right in occupied)

    for pattern in (_URL_RE, _FILE_URL_RE, _PATH_RE):
        for match in pattern.finditer(text):
            if protected(match.start()):
                continue
            if item := _candidate(match.group(0), None, origin):
                candidates.append(item)

    return candidates


def _content_texts(contents: Iterable[object]) -> Iterable[str]:
    for content in contents:
        if text := get_text(content):
            yield text


def _message_texts(message, index: int) -> Iterable[tuple[str, str]]:
    role = getattr(message, "role", "message")
    for text in _content_texts(getattr(message, "content", ())):
        yield text, f"{role} message {index}"

    for tool_call in (getattr(message, "tool_calls", None) or {}).values():
        params = getattr(tool_call, "params", None)
        name = getattr(params, "name", "tool")
        arguments = getattr(params, "arguments", None)
        if arguments:
            yield str(arguments), f"tool call {name}"

    for tool_result in (getattr(message, "tool_results", None) or {}).values():
        for text in _content_texts(getattr(tool_result, "content", ())):
            yield text, f"tool result {index}"


def _recent_messages(ctx, *, last_turn: bool):
    history = list(ctx.message_history)
    if not last_turn:
        return history
    for index in range(len(history) - 1, -1, -1):
        if getattr(history[index], "role", None) == "user":
            return history[index + 1 :]
    return history


def _scan(ctx, *, last_turn: bool) -> list[ImageCandidate]:
    seen: set[str] = set()
    items: list[ImageCandidate] = []
    messages = _recent_messages(ctx, last_turn=last_turn)
    base_index = len(ctx.message_history) - len(messages)
    for offset, message in enumerate(messages):
        for text, origin in _message_texts(message, base_index + offset + 1):
            for item in _extract_from_text(text, origin):
                if item.source in seen:
                    continue
                seen.add(item.source)
                items.append(item)
    return items


def _format_item(index: int, item: ImageCandidate) -> str:
    label = item.label or item.source
    return f"{index}. {label}\n   {item.source}\n   _{item.origin}_"


def _list_markdown(items: list[ImageCandidate], *, title: str) -> str:
    if not items:
        return f"{title}\n\nNo image URLs or paths found."
    lines = [title, ""]
    lines.extend(_format_item(index, item) for index, item in enumerate(items, 1))
    return "\n\n".join(lines)


def _direct_source(selector: str) -> ImageCandidate | None:
    if _looks_like_image_source(selector):
        return ImageCandidate(source=_clean_source(selector), label=None, origin="argument")
    return None


def _select(items: list[ImageCandidate], selector: str) -> list[ImageCandidate]:
    normalized = selector.strip()
    if not normalized or normalized == "last":
        return items[-1:] if items else []
    if normalized == "all":
        return items
    if normalized.isdecimal():
        index = int(normalized)
        return [items[index - 1]] if 1 <= index <= len(items) else []
    if direct := _direct_source(normalized):
        return [direct]
    lowered = normalized.lower()
    return [item for item in items if lowered in (item.label or "").lower() or lowered in item.source.lower()]


def _quote_if_needed(value: str) -> str:
    return shlex.quote(value) if any(ch.isspace() for ch in value) else value


async def images(ctx):
    try:
        parts = shlex.split(ctx.arguments)
    except ValueError as exc:
        return PluginCommandActionResult(message=f"Invalid /images arguments: {exc}")

    action = parts[0].lower() if parts else "last"
    selector = " ".join(parts[1:]).strip()

    if action in {"list", "ls"}:
        return PluginCommandActionResult(
            markdown=_list_markdown(_scan(ctx, last_turn=False), title="Recent image sources")
        )

    if action == "last":
        items = _scan(ctx, last_turn=True)
        if not selector:
            return PluginCommandActionResult(
                markdown=_list_markdown(items, title="Image sources since the last user turn")
            )
        selected = _select(items, selector)
        return _show(selected, selector=selector, scope="last user turn")

    if action in {"show", "view", "open"}:
        items = _scan(ctx, last_turn=False)
        selected = _select(items, selector or "last")
        return _show(selected, selector=selector or "last", scope="recent history")

    if direct := _direct_source(action):
        return _show([direct], selector=action, scope="argument")

    return PluginCommandActionResult(
        markdown=(
            "Usage: `/images [list|last|show] [all|last|number|source]`\n"
            "`/images` (no args) lists images from the last turn; `list` shows all.\n\n"
            "Examples:\n"
            "- `/images`\n"
            "- `/images list`\n"
            "- `/images last all`\n"
            "- `/images show 1`\n"
            "- `/images show /tmp/output.png`"
        )
    )


def _show(items: list[ImageCandidate], *, selector: str, scope: str) -> PluginCommandActionResult:
    if not items:
        return PluginCommandActionResult(message=f"No image matched `{selector}` in {scope}.")
    markdown = "\n".join(f"Showing `{item.label or item.source}`: {item.source}" for item in items)
    return PluginCommandActionResult(
        markdown=markdown,
        images=[
            PluginCommandActionImage(source=item.source, label=item.label or item.source)
            for item in items
        ],
    )


async def complete_images(ctx):
    tokens = list(ctx.completed_tokens)
    if not tokens:
        return [
            PluginCommandCompletion("list", detail="list recent image URLs/paths"),
            PluginCommandCompletion("last", detail="scan since the last user turn"),
            PluginCommandCompletion("show", detail="render an image by index, last, path, or URL"),
        ]

    action = tokens[0].lower()
    if action == "last":
        return _selector_completions(_scan(ctx, last_turn=True), include_all=True)
    if action in {"show", "view", "open"}:
        return _selector_completions(_scan(ctx, last_turn=False), include_all=True)
    return []


def _selector_completions(items: list[ImageCandidate], *, include_all: bool) -> list[PluginCommandCompletion]:
    completions = [PluginCommandCompletion("last", detail="newest discovered image")]
    if include_all:
        completions.append(PluginCommandCompletion("all", detail="render all discovered images"))
    for index, item in enumerate(items, 1):
        label = item.label or item.source
        completions.append(
            PluginCommandCompletion(
                str(index),
                display=f"{index} {label}",
                detail=item.origin,
            )
        )
    return completions[:30]
