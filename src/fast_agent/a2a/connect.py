"""Helpers for interactive A2A connection requests."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

_TRANSPORT_ALIASES = {
    "jsonrpc": "JSONRPC",
    "json-rpc": "JSONRPC",
    "rpc": "JSONRPC",
    "http": "HTTP+JSON",
    "http+json": "HTTP+JSON",
    "rest": "HTTP+JSON",
    "grpc": "GRPC",
}


@dataclass(frozen=True, slots=True)
class A2AConnectRequest:
    url: str
    name: str | None = None
    transport: str | None = None
    relative_card_path: str | None = None


def parse_a2a_connect_arguments(arguments: str | None) -> tuple[A2AConnectRequest | None, str | None]:
    if not arguments:
        return None, "Usage: /a2a connect <base-url-or-card-url> [--transport JSONRPC|HTTP+JSON|GRPC] [--name NAME] [--card-path PATH]"
    try:
        tokens = shlex.split(arguments)
    except ValueError as exc:
        return None, str(exc)

    url: str | None = None
    name: str | None = None
    transport: str | None = None
    card_path: str | None = None
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in {"--transport", "-t", "--name", "--card-path"}:
            if index + 1 >= len(tokens):
                return None, f"{token} requires a value"
            value = tokens[index + 1]
            if token in {"--transport", "-t"}:
                transport = normalize_a2a_transport(value)
                if transport is None:
                    return None, f"Unsupported A2A transport: {value}"
            elif token == "--name":
                name = _normalize_agent_name(value)
                if not name:
                    return None, f"Invalid agent name: {value}"
            else:
                card_path = value
            index += 2
            continue
        if token.startswith("-"):
            return None, f"Unknown /a2a connect option: {token}"
        if url is not None:
            return None, f"Unexpected /a2a connect argument: {token}"
        url = token
        index += 1

    if url is None:
        return None, "A2A base URL or agent-card URL is required"
    normalized_url, inferred_card_path, error = normalize_a2a_url(url)
    if error:
        return None, error
    return (
        A2AConnectRequest(
            url=normalized_url,
            name=name,
            transport=transport,
            relative_card_path=card_path or inferred_card_path,
        ),
        None,
    )


def normalize_a2a_transport(value: str) -> str | None:
    return _TRANSPORT_ALIASES.get(value.strip().lower())


def normalize_a2a_url(url: str) -> tuple[str, str | None, str | None]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "", None, "A2A connect expects an http(s) base URL or agent-card URL"
    path = parsed.path or ""
    if path.endswith("agent-card.json"):
        base = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        relative_path = path if path.startswith("/") else f"/{path}"
        return base, relative_path, None
    return url.rstrip("/"), None, None


def _normalize_agent_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_")
