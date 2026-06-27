"""Parse and resolve inline MCP resource mentions for interactive prompt input."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from urllib.parse import quote

from mcp.types import ContentBlock, EmbeddedResource, ReadResourceResult, TextContent

from fast_agent.mcp.helpers.content_helpers import image_link, resource_link
from fast_agent.mcp.mcp_content import MCPFile, MCPImage
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
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
_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")
_TEMPLATE_ARG_KEY_RE = re.compile(r"(?:^|,)(?P<key>[A-Za-z0-9_.-]+)=")
_TEMPLATE_ARGS_RE = re.compile(
    r"^(?P<template>[^{]*)\{(?P<args>[A-Za-z0-9_.-]+=[^{}]*(?:,[A-Za-z0-9_.-]+=[^{}]*)*)\}$"
)
_TEMPLATE_OPERATORS = "+#./;?&"


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
    mentions: list[ParsedMention]
    warnings: list[str]


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


@dataclass(frozen=True)
class _TemplateVarSpec:
    name: str
    explode: bool
    prefix: int | None


@dataclass(frozen=True, slots=True)
class _ParsedTemplateArgs:
    template_uri: str
    args: dict[str, str]


@dataclass(frozen=True, slots=True)
class _TemplateOperatorStyle:
    prefix: str = ""
    separator: str = ","
    named: bool = False
    if_empty: str = ""


_TEMPLATE_OPERATOR_STYLES: dict[str, _TemplateOperatorStyle] = {
    "#": _TemplateOperatorStyle(prefix="#"),
    ".": _TemplateOperatorStyle(prefix=".", separator="."),
    "/": _TemplateOperatorStyle(prefix="/", separator="/"),
    ";": _TemplateOperatorStyle(prefix=";", separator=";", named=True),
    "?": _TemplateOperatorStyle(prefix="?", separator="&", named=True, if_empty="="),
    "&": _TemplateOperatorStyle(prefix="&", separator="&", named=True, if_empty="="),
}


def _parse_template_varspec(raw_spec: str) -> _TemplateVarSpec | None:
    spec = raw_spec.strip()
    if not spec:
        return None

    explode = spec.endswith("*")
    if explode:
        spec = spec[:-1]

    prefix: int | None = None
    if ":" in spec:
        var_name, prefix_str = spec.split(":", 1)
        var_name = var_name.strip()
        prefix = int(prefix_str) if prefix_str.isdigit() else None
    else:
        var_name = spec.strip()

    if not var_name:
        return None

    return _TemplateVarSpec(name=var_name, explode=explode, prefix=prefix)


def _iter_template_varspecs(expression_body: str) -> list[_TemplateVarSpec]:
    if not expression_body:
        return []

    body = expression_body
    if body[0] in _TEMPLATE_OPERATORS:
        body = body[1:]

    specs: list[_TemplateVarSpec] = []
    for raw_spec in body.split(","):
        parsed = _parse_template_varspec(raw_spec)
        if parsed is None:
            continue
        specs.append(parsed)
    return specs


def _encode_template_value(
    value: str,
    *,
    allow_reserved: bool,
    preserve_slashes: bool,
) -> str:
    safe_chars = "%"
    if allow_reserved:
        safe_chars += "/:?#[]@!$&'()*+,;="
    if preserve_slashes and "/" not in safe_chars:
        safe_chars += "/"
    return quote(value, safe=safe_chars)


def _expand_template_expression(expression_body: str, args: dict[str, str]) -> str:
    if not expression_body:
        return ""

    operator = ""
    body = expression_body
    if body[0] in _TEMPLATE_OPERATORS:
        operator = body[0]
        body = body[1:]

    varspecs = _iter_template_varspecs(body)
    if not varspecs:
        return ""

    style = _TEMPLATE_OPERATOR_STYLES.get(operator, _TemplateOperatorStyle())
    allow_reserved = operator in {"+", "#"}

    expanded_parts: list[str] = []
    for spec in varspecs:
        value = args.get(spec.name, "")
        if spec.prefix is not None:
            value = value[: spec.prefix]

        encoded_value = _encode_template_value(
            value,
            allow_reserved=allow_reserved,
            # Preserve legacy behavior for simple `{var}` path-like values.
            preserve_slashes=spec.explode or operator == "",
        )

        if style.named:
            if encoded_value:
                expanded_parts.append(f"{spec.name}={encoded_value}")
            else:
                expanded_parts.append(f"{spec.name}{style.if_empty}")
        else:
            expanded_parts.append(encoded_value)

    if not expanded_parts:
        return ""
    return style.prefix + style.separator.join(expanded_parts)


def template_argument_names(template_uri: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    for raw_expression in _PLACEHOLDER_RE.findall(template_uri):
        for spec in _iter_template_varspecs(raw_expression):
            if spec.name in seen:
                continue
            seen.add(spec.name)
            names.append(spec.name)

    return names


def _template_arg_pairs(args_str: str) -> list[tuple[str, str]]:
    matches = list(_TEMPLATE_ARG_KEY_RE.finditer(args_str))
    pairs: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        value_start = match.end()
        value_end = matches[index + 1].start() if index + 1 < len(matches) else len(args_str)
        pairs.append((match.group("key").strip(), args_str[value_start:value_end].strip()))
    return pairs


def _parse_template_args(value: str) -> _ParsedTemplateArgs:
    match = _TEMPLATE_ARGS_RE.match(value)
    if not match:
        return _ParsedTemplateArgs(template_uri=value, args={})

    template_uri = match.group("template")
    args_str = match.group("args")
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

    def _replace(match: re.Match[str]) -> str:
        expression_body = match.group(1).strip()
        return _expand_template_expression(expression_body, args)

    return _PLACEHOLDER_RE.sub(_replace, template_uri)


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


def parse_mentions(text: str, *, cwd: Path | None = None) -> ParsedMentions:
    """Parse supported resource mentions from text and strip them from the sent message body."""
    mentions: list[ParsedMention] = []
    warnings: list[str] = []
    pieces: list[str] = []
    cursor = 0

    for match in _TOKEN_RE.finditer(text):
        token = match.group("token")
        token_start = match.start("token")
        token_end = match.end("token")

        pieces.append(text[cursor:token_start])

        parsed: ParsedMention | None
        try:
            parsed = _parse_token(token, start=token_start, end=token_end, cwd=cwd)
        except (ResourceMentionError, ValueError) as exc:
            parsed = None
            warnings.append(f"Malformed resource mention '{token}': {exc}")

        if parsed is None:
            pieces.append(token)
        else:
            mentions.append(parsed)

        cursor = token_end

    pieces.append(text[cursor:])

    cleaned_text = "".join(pieces)
    cleaned_text = re.sub(r"[ \t]{2,}", " ", cleaned_text).strip()

    return ParsedMentions(
        text=text,
        cleaned_text=cleaned_text,
        mentions=_dedupe_mentions(mentions),
        warnings=warnings,
    )


def _dedupe_mentions(mentions: Sequence[ParsedMention]) -> list[ParsedMention]:
    return unique_preserve_order(
        mentions,
        key=lambda mention: (mention.server_name, mention.resource_uri),
    )


async def resolve_mentions(agent: Any, parsed: ParsedMentions) -> ResolvedMentions:
    """Resolve parsed mentions to embedded MCP resource blocks."""
    if not parsed.mentions:
        return ResolvedMentions(
            text=parsed.text,
            cleaned_text=parsed.cleaned_text,
            mentions=[],
            resources=[],
        )

    remote_mentions = [
        mention
        for mention in parsed.mentions
        if mention.server_name not in {FILE_MENTION_SERVER, URL_MENTION_SERVER}
    ]
    resource_agent = _as_resource_mention_resolver(agent)
    if remote_mentions and resource_agent is None:
        raise ResourceMentionError("Current agent does not support MCP resources")

    resources: list[ContentBlock] = []
    failures: list[str] = []
    for mention in parsed.mentions:
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
            failures.append(f"{mention.raw}: {exc}")

    if failures:
        raise ResourceMentionError("; ".join(failures))

    return ResolvedMentions(
        text=parsed.text,
        cleaned_text=parsed.cleaned_text,
        mentions=list(parsed.mentions),
        resources=resources,
    )


def build_prompt_with_resources(
    original_text: str,
    resolved: ResolvedMentions,
) -> PromptMessageExtended:
    """Build PromptMessageExtended with text content and embedded resources."""
    text = resolved.cleaned_text if resolved.mentions else original_text
    text_content = TextContent(type="text", text=text)
    text_meta = dict(getattr(text_content, "meta", None) or {})
    text_meta["fast_agent_original_text"] = original_text
    text_content.meta = text_meta
    content: list[ContentBlock] = [text_content]
    content.extend(resolved.resources)
    return PromptMessageExtended(role="user", content=content)


def _resolve_local_content_block(path_text: str) -> ContentBlock:
    path = Path(path_text)
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(path)

    message = MCPImage(path=path) if _is_image_path(path) else MCPFile(path=path)
    content = message["content"]
    meta = dict(getattr(content, "meta", None) or {})
    meta["fast_agent_source_uri"] = path.as_uri()
    content.meta = meta
    return content


def _resolve_remote_content_block(url: str) -> ContentBlock:
    inferred = resource_link(url)
    mime_type = inferred.mimeType or "application/octet-stream"
    if mime_type.startswith("image/"):
        return image_link(url, mime_type=mime_type)
    return inferred


def _is_image_path(path: Path) -> bool:
    from fast_agent.mcp.mime_utils import guess_mime_type, is_image_mime_type

    return is_image_mime_type(guess_mime_type(str(path)))
