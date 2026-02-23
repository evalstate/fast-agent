"""Parse and resolve inline MCP resource mentions for interactive prompt input."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import quote

from mcp.types import EmbeddedResource, ReadResourceResult, TextContent

from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

if TYPE_CHECKING:
    from collections.abc import Sequence


_TOKEN_RE = re.compile(r"(?P<prefix>^|\s)(?P<token>\^[^\s]+)")
_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")
_TEMPLATE_ARGS_RE = re.compile(
    r"^(?P<template>.+)\{(?P<args>[A-Za-z0-9_.-]+=[^{}]*(?:,[A-Za-z0-9_.-]+=[^{}]*)*)\}$"
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
    resources: list[EmbeddedResource]


class ResourceMentionError(ValueError):
    """Raised when one or more resource mentions cannot be resolved."""


@dataclass(frozen=True)
class _TemplateVarSpec:
    name: str
    explode: bool
    prefix: int | None


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
        if prefix_str.isdigit():
            prefix = int(prefix_str)
        else:
            prefix = None
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

    separator = ","
    prefix = ""
    named = False
    if_empty = ""
    allow_reserved = operator in {"+", "#"}

    if operator == "#":
        prefix = "#"
    elif operator == ".":
        prefix, separator = ".", "."
    elif operator == "/":
        prefix, separator = "/", "/"
    elif operator == ";":
        prefix, separator, named = ";", ";", True
    elif operator == "?":
        prefix, separator, named, if_empty = "?", "&", True, "="
    elif operator == "&":
        prefix, separator, named, if_empty = "&", "&", True, "="

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

        if named:
            if encoded_value:
                expanded_parts.append(f"{spec.name}={encoded_value}")
            else:
                expanded_parts.append(f"{spec.name}{if_empty}")
        else:
            expanded_parts.append(encoded_value)

    if not expanded_parts:
        return ""
    return prefix + separator.join(expanded_parts)


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



def _parse_template_args(value: str) -> tuple[str, dict[str, str]]:
    match = _TEMPLATE_ARGS_RE.match(value)
    if not match:
        return value, {}

    template_uri = match.group("template")
    args_str = match.group("args")
    args: dict[str, str] = {}
    for raw_pair in args_str.split(","):
        key, raw_value = raw_pair.split("=", 1)
        args[key.strip()] = raw_value.strip()
    return template_uri, args


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


def _parse_token(token: str, *, start: int, end: int) -> ParsedMention | None:
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

    template_uri, args = _parse_template_args(resource_expr)
    resource_uri = _render_template_uri(template_uri, args)

    return ParsedMention(
        raw=token,
        server_name=server_name,
        resource_uri=resource_uri,
        start=start,
        end=end,
    )


def parse_mentions(text: str) -> ParsedMentions:
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
            parsed = _parse_token(token, start=token_start, end=token_end)
        except ResourceMentionError as exc:
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

    unique_mentions: list[ParsedMention] = []
    seen: set[tuple[str, str]] = set()
    for mention in mentions:
        key = (mention.server_name, mention.resource_uri)
        if key in seen:
            continue
        seen.add(key)
        unique_mentions.append(mention)

    return ParsedMentions(
        text=text,
        cleaned_text=cleaned_text,
        mentions=unique_mentions,
        warnings=warnings,
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

    get_resource = getattr(agent, "get_resource", None)
    if not callable(get_resource):
        raise ResourceMentionError("Current agent does not support MCP resources")

    resources: list[EmbeddedResource] = []
    failures: list[str] = []
    for mention in parsed.mentions:
        try:
            result: ReadResourceResult = await get_resource(
                mention.resource_uri,
                namespace=mention.server_name,
            )
            for content in result.contents:
                resources.append(EmbeddedResource(type="resource", resource=content, annotations=None))
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
    content = [TextContent(type="text", text=text)]
    content.extend(resolved.resources)
    return PromptMessageExtended(role="user", content=content)


def mentions_in_text(text: str) -> Sequence[ParsedMention]:
    """Convenience helper primarily for tests."""
    return parse_mentions(text).mentions
