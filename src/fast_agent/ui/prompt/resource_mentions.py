"""Parse and resolve inline attachment mentions for interactive prompt input."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from mcp_types import ContentBlock, EmbeddedResource, ReadResourceResult, TextContent
from uritemplate import URITemplate

from fast_agent.mcp.helpers.content_helpers import image_link, resource_link
from fast_agent.mcp.mcp_content import MCPFile
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.ptc.mentions import (
    ORIGINAL_TEXT_META,
    TOOL_MENTION_SERVER,
    ToolMentionError,
    iter_tool_mentions,
    mention_name,
    resolve_tool_mention_surface,
    rewrite_tool_mentions,
    tool_mention_names,
    tool_mention_token,
)
from fast_agent.tools.attach_media import build_attach_media
from fast_agent.ui.prompt.attachment_tokens import (
    FILE_MENTION_SERVER,
    URL_MENTION_SERVER,
    normalize_local_attachment_reference,
    normalize_remote_attachment_reference,
)
from fast_agent.utils.collections import unique_preserve_order

if TYPE_CHECKING:
    from collections.abc import Sequence


_TOKEN_RE = re.compile(r"(?P<prefix>^|\s)(?P<token>\^[^\s]+)")
_TEMPLATE_ARG_KEY_RE = re.compile(r"(?:^|,)(?P<key>[A-Za-z0-9_.-]+)=")
_LEADING_BLANK_LINES_RE = re.compile(r"\A(?:[ \t]*\n)+")
_SLASH_SENTINEL = "__fast_agent_slash__"


@dataclass(frozen=True)
class ParsedMention:
    raw: str
    server_name: str
    resource_uri: str
    start: int
    end: int


@dataclass(frozen=True)
class ParsedMentions:
    text: str
    cleaned_text: str
    """``text`` without attachment and resource tokens. ``^tool:`` tokens stay in place until
    ``resolve_mentions`` rewrites them against the agent's PTC functions."""
    mentions: list[ParsedMention]
    warnings: list[str]
    cwd: Path | None = None
    tool_mentions: bool = True
    """Whether ``^tool:`` names PTC functions rather than resources of an MCP server ``tool``."""


@dataclass(frozen=True)
class ResolvedMentions:
    text: str
    cleaned_text: str
    mentions: list[ParsedMention]
    resources: list[ContentBlock]


class ResourceMentionError(ValueError):
    """Raised when one or more resource mentions cannot be resolved."""


@runtime_checkable
class _ResourceMentionResolver(Protocol):
    async def get_resource(
        self,
        resource_uri: str,
        namespace: str | None = None,
    ) -> ReadResourceResult: ...


def _as_resource_mention_resolver(agent: Any) -> _ResourceMentionResolver | None:
    if not isinstance(agent, _ResourceMentionResolver):
        return None
    if not callable(agent.get_resource):
        return None
    return agent


@dataclass(frozen=True, slots=True)
class _ParsedTemplateArgs:
    template_uri: str
    args: dict[str, str]


def template_argument_names(template_uri: str) -> list[str]:
    return list(URITemplate(template_uri).variable_names)


def _template_arg_pairs(args_str: str) -> list[tuple[str, str]]:
    matches = list(_TEMPLATE_ARG_KEY_RE.finditer(args_str))
    pairs: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(args_str)
        pairs.append((match.group("key").strip(), args_str[value_start:value_end].strip()))
    return pairs


def _looks_like_template_args(args_str: str) -> bool:
    matches = list(_TEMPLATE_ARG_KEY_RE.finditer(args_str))
    if not matches or matches[0].start() != 0:
        return False
    return all(match.group("key").strip() for match in matches)


def _parse_template_args(value: str) -> _ParsedTemplateArgs:
    if not value.endswith("}"):
        return _ParsedTemplateArgs(template_uri=value, args={})

    args_start = value.rfind("{")
    if args_start < 0:
        return _ParsedTemplateArgs(template_uri=value, args={})

    args_str = value[args_start + 1 : -1]
    if not _looks_like_template_args(args_str):
        return _ParsedTemplateArgs(template_uri=value, args={})

    template_uri = value[:args_start]
    args = dict(_template_arg_pairs(args_str))
    return _ParsedTemplateArgs(template_uri=template_uri, args=args)


def _render_template_uri(template_uri: str, args: dict[str, str]) -> str:
    required_names = template_argument_names(template_uri)
    if not required_names:
        return template_uri

    missing = [name for name in required_names if name not in args]
    if missing:
        missing_str = ", ".join(sorted(set(missing)))
        raise ResourceMentionError(f"Missing template arguments: {missing_str}")

    template = URITemplate(template_uri)
    expansion_args: dict[str, Any] = dict(args)
    for variable in template.variables:
        operator = getattr(variable.operator, "value", "")
        for name, options in variable.variables:
            value = args.get(name)
            if value is None:
                continue
            if operator == "/" and options.get("explode") and "/" in value:
                expansion_args[name] = value.split("/")
            elif operator == "" and options.get("prefix") is None and "/" in value:
                expansion_args[name] = value.replace("/", _SLASH_SENTINEL)

    return template.expand(expansion_args).replace(_SLASH_SENTINEL, "/")


def _parse_token(
    token: str,
    *,
    start: int,
    end: int,
    cwd: Path | None = None,
) -> ParsedMention | None:
    if not token.startswith("^"):
        return None

    payload = token[1:]
    if ":" not in payload:
        return None

    server_name, resource_expr = payload.split(":", 1)
    server_name = server_name.strip()
    resource_expr = resource_expr.strip()
    if not server_name or not resource_expr:
        return None

    if server_name == FILE_MENTION_SERVER:
        resource_uri = str(normalize_local_attachment_reference(resource_expr, cwd=cwd))
    elif server_name == URL_MENTION_SERVER:
        resource_uri = normalize_remote_attachment_reference(resource_expr)
    else:
        template_args = _parse_template_args(resource_expr)
        resource_uri = _render_template_uri(template_args.template_uri, template_args.args)

    return ParsedMention(
        raw=token,
        server_name=server_name,
        resource_uri=resource_uri,
        start=start,
        end=end,
    )


def parse_mentions(
    text: str,
    *,
    cwd: Path | None = None,
    tool_mentions: bool = True,
) -> ParsedMentions:
    """Parse supported mentions from text and strip resource tokens from the sent message body.

    With ``tool_mentions``, ``^tool:<name>`` tokens are PTC function mentions and stay in the
    text; otherwise they are resource mentions for an MCP server named ``tool``.
    """
    mentions: list[ParsedMention] = []
    warnings: list[str] = []
    removed: list[tuple[int, int]] = []
    tool_prefix = tool_mention_token("")

    if tool_mentions:
        mentions.extend(
            ParsedMention(
                raw=match.group(),
                server_name=TOOL_MENTION_SERVER,
                resource_uri=mention_name(match),
                start=match.start(),
                end=match.end(),
            )
            for match in iter_tool_mentions(text)
        )
    tool_starts = {mention.start for mention in mentions}

    for match in _TOKEN_RE.finditer(text):
        token = match.group("token")
        token_start = match.start("token")
        token_end = match.end("token")
        if tool_mentions and token.startswith(tool_prefix):
            if token_start not in tool_starts:
                # Not a function name: an error for PTC agents, a resource URI otherwise.
                mentions.append(
                    ParsedMention(
                        raw=token,
                        server_name=TOOL_MENTION_SERVER,
                        resource_uri=token.removeprefix(tool_prefix),
                        start=token_start,
                        end=token_end,
                    )
                )
            continue

        parsed: ParsedMention | None
        try:
            parsed = _parse_token(token, start=token_start, end=token_end, cwd=cwd)
        except (ResourceMentionError, ValueError) as exc:
            parsed = None
            warnings.append(f"Malformed resource mention '{token}': {exc}")

        if parsed is not None:
            mentions.append(parsed)
            removed.append((token_start, token_end))

    mentions.sort(key=lambda mention: mention.start)
    return ParsedMentions(
        text=text,
        cleaned_text=_remove_tokens(text, removed),
        mentions=_dedupe_mentions(mentions),
        warnings=warnings,
        cwd=cwd,
        tool_mentions=tool_mentions,
    )


def _remove_tokens(text: str, spans: Sequence[tuple[int, int]]) -> str:
    """Remove token spans, collapsing only the spaces next to each token.

    Indentation and spacing elsewhere are preserved byte for byte.
    """
    if not spans:
        return text
    for start, end in reversed(spans):
        after = end
        while after < len(text) and text[after] in " \t":
            after += 1
        if after == len(text) or text[after] == "\n":
            while start and text[start - 1] in " \t":
                start -= 1
        text = text[:start] + text[after:]
    return _LEADING_BLANK_LINES_RE.sub("", text).rstrip()


def _dedupe_mentions(mentions: Sequence[ParsedMention]) -> list[ParsedMention]:
    return unique_preserve_order(
        mentions,
        key=lambda mention: (mention.server_name, mention.resource_uri),
    )


async def resolve_mentions(agent: Any, parsed: ParsedMentions) -> ResolvedMentions:
    """Resolve parsed mentions to content blocks.

    ``^tool:`` mentions name PTC functions only when PTC is enabled for ``agent``; otherwise
    they are resolved as resources of an MCP server named ``tool``. Every mention is resolved
    before failing, so one ``ResourceMentionError`` reports all problems.
    """
    if not parsed.mentions:
        return ResolvedMentions(
            text=parsed.text,
            cleaned_text=parsed.cleaned_text,
            mentions=[],
            resources=[],
        )

    cleaned_text = parsed.cleaned_text
    resources: list[ContentBlock] = []
    failures: list[str] = []
    if parsed.tool_mentions and any(
        mention.server_name == TOOL_MENTION_SERVER for mention in parsed.mentions
    ):
        surface = await resolve_tool_mention_surface(agent)
        if surface.programmatic is None:
            reparsed = parse_mentions(parsed.text, cwd=parsed.cwd, tool_mentions=False)
            return await resolve_mentions(agent, reparsed)
        names = set(tool_mention_names(cleaned_text))
        failures.extend(
            f"{mention.raw}: not a `{surface.programmatic.tool_name}` function name"
            for mention in parsed.mentions
            if mention.server_name == TOOL_MENTION_SERVER and mention.resource_uri not in names
        )
        try:
            rewrite = rewrite_tool_mentions(surface, [cleaned_text])
        except ToolMentionError as exc:
            failures.append(str(exc))
        else:
            [cleaned_text] = rewrite.texts
            if rewrite.definitions is not None:
                resources.append(rewrite.definitions)

    resource_agent = _as_resource_mention_resolver(agent)
    for mention in parsed.mentions:
        if parsed.tool_mentions and mention.server_name == TOOL_MENTION_SERVER:
            continue
        try:
            if mention.server_name == FILE_MENTION_SERVER:
                resources.append(_resolve_local_content_block(mention.resource_uri))
                continue
            if mention.server_name == URL_MENTION_SERVER:
                resources.append(_resolve_remote_content_block(mention.resource_uri))
                continue
            if resource_agent is None:
                raise ResourceMentionError("Current agent does not support MCP resources")
            result = await resource_agent.get_resource(
                mention.resource_uri,
                namespace=mention.server_name,
            )
            resources.extend(
                EmbeddedResource(type="resource", resource=content, annotations=None)
                for content in result.contents
            )
        except Exception as exc:
            if mention.server_name == TOOL_MENTION_SERVER:
                failures.append(
                    f"{mention.raw}: programmatic tool calling is not enabled for this agent "
                    f"and no MCP resource matched ({exc})"
                )
            else:
                failures.append(f"{mention.raw}: {exc}")

    if failures:
        raise ResourceMentionError("; ".join(failures))

    return ResolvedMentions(
        text=parsed.text,
        cleaned_text=cleaned_text,
        mentions=list(parsed.mentions),
        resources=resources,
    )


def build_prompt_with_resources(
    original_text: str,
    resolved: ResolvedMentions,
) -> PromptMessageExtended:
    """Build PromptMessageExtended with text content and embedded resources."""
    text = resolved.cleaned_text if resolved.mentions else original_text
    text_content = TextContent(type="text", text=text, meta={ORIGINAL_TEXT_META: original_text})
    content: list[ContentBlock] = [text_content]
    content.extend(resolved.resources)
    return PromptMessageExtended(role="user", content=content)


def _resolve_local_content_block(path_text: str) -> ContentBlock:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(path)

    if _is_image_path(path):
        content = build_attach_media(str(path), base_directory=path.parent).block
    else:
        content = MCPFile(path=path)["content"]
    meta = dict(getattr(content, "meta", None) or {})
    meta["fast_agent_source_uri"] = path.as_uri()
    content.meta = meta
    return content


def _resolve_remote_content_block(url: str) -> ContentBlock:
    inferred = resource_link(url)
    mime_type = inferred.mime_type or "application/octet-stream"
    if mime_type.startswith("image/"):
        return image_link(url, mime_type=mime_type)
    return inferred


def _is_image_path(path: Path) -> bool:
    from PIL import Image

    from fast_agent.mcp.mime_utils import guess_mime_type, is_image_mime_type

    mime_type = guess_mime_type(str(path))
    if is_image_mime_type(mime_type):
        return True
    if mime_type != "application/octet-stream":
        return False
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, SyntaxError, ValueError):
        return False
    return True
