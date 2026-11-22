"""``^tool:<name>`` mentions: add PTC function definitions to a user message.

A mention neither calls a tool nor grants access to one. It inserts the same canonical stub that
``search_tools`` returns, so the model can call a function whose definition is deferred from the
``exec_sandbox`` description without searching for it first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

from mcp_types import TextContent

from fast_agent.tools.tool_surface import SurfaceTool, ToolSurface, ToolSurfaceProvider

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from fast_agent.mcp.prompt_message_extended import PromptMessageExtended

TOOL_MENTION_SERVER: Final = "tool"
"""Mention "server" name reserved for PTC function definitions: ``^tool:<name>``."""
TOOL_DEFINITIONS_META: Final = "fast_agent_tool_definitions"
"""Text block metadata key listing the Python names of inserted definitions."""
TOOL_DEFINITIONS_TAG: Final = "fastagent:tool-definitions"
ORIGINAL_TEXT_META: Final = "fast_agent_original_text"
"""Text block metadata key holding the text as typed, before mentions were rewritten."""
TOOL_MENTION_PATTERN: Final = (
    rf"\^{TOOL_MENTION_SERVER}:"
    r"(?:[A-Za-z_](?:[A-Za-z0-9_.-]*[A-Za-z0-9_-])?)(?=$|\s|[,.):;?!])"
)
"""A whole ``^tool:<name>`` token. Names may be namespaced (``server__tool-name.v2``); trailing
sentence punctuation (``, . ) : ; ? !``) ends the token instead of being part of the name."""
_TOOL_MENTION_RE = re.compile(rf"(?<!\S){TOOL_MENTION_PATTERN}")
_TOOL_MENTION_PREFIX_LEN: Final = len(TOOL_MENTION_SERVER) + 2


@runtime_checkable
class ToolMentionNamesProvider(Protocol):
    """Agents that know their ``^tool:`` names without an async tool listing."""

    @property
    def ptc_mention_names(self) -> frozenset[str] | None:
        """Python and tool names of the latest catalogue; None when unknown or PTC is off."""
        ...


class ToolMentionError(ValueError):
    """A ``^tool:`` mention that does not name a current PTC function."""


@dataclass(frozen=True, slots=True)
class ToolMentionRewrite:
    texts: tuple[str, ...]
    """The input texts with mentions rewritten (see ``rewrite_tool_mentions``)."""
    definitions: TextContent | None
    """One ``<fastagent:tool-definitions>`` block for every mentioned function; None without
    mentions."""


def tool_mention_token(name: str) -> str:
    return f"^{TOOL_MENTION_SERVER}:{name}"


def iter_tool_mentions(text: str) -> Iterator[re.Match[str]]:
    """Whitespace-delimited ``^tool:`` tokens; ``mention_name(match)`` is the mentioned name."""
    return _TOOL_MENTION_RE.finditer(text)


def mention_name(match: re.Match[str]) -> str:
    return match.group()[_TOOL_MENTION_PREFIX_LEN:]


def tool_mention_names(text: str) -> list[str]:
    """Return names from whitespace-delimited tool mentions in plain text."""
    return [mention_name(match) for match in iter_tool_mentions(text)]


async def resolve_tool_mention_surface(agent: object) -> ToolSurface:
    """The agent's tool surface; empty for agents without programmatic functions."""
    if isinstance(agent, ToolSurfaceProvider):
        return await agent.tool_surface()
    return ToolSurface(tools=())


def find_programmatic_function(surface: ToolSurface, name: str) -> SurfaceTool | None:
    """Find a PTC function by Python name, then by tool name.

    A retired Python name never falls back to a different function whose tool is named alike.
    """
    functions = surface.programmatic_functions()
    function = next((tool for tool in functions if tool.python_name == name), None)
    if function is not None or _is_retired(surface, name):
        return function
    return next((tool for tool in functions if tool.tool.name == name), None)


def _is_retired(surface: ToolSurface, name: str) -> bool:
    return surface.programmatic is not None and name in surface.programmatic.retired


def _resolve_mentioned_functions(
    surface: ToolSurface, names: Iterable[str]
) -> dict[str, SurfaceTool]:
    """Map each mentioned name to its PTC function, in mention order."""
    if surface.programmatic is None:
        first = next(iter(names), "")
        raise ToolMentionError(
            f"^{TOOL_MENTION_SERVER}:{first}: programmatic tool calling is not enabled "
            "for this agent."
        )
    resolved: dict[str, SurfaceTool] = {}
    for name in names:
        function = find_programmatic_function(surface, name)
        if function is None or function.python_name is None:
            if _is_retired(surface, name):
                raise ToolMentionError(
                    f"^{TOOL_MENTION_SERVER}:{name}: this function is no longer available; "
                    "its tool was removed. Use /tools to list available functions."
                )
            raise ToolMentionError(
                f"^{TOOL_MENTION_SERVER}:{name}: no `{surface.programmatic.tool_name}` "
                "function has this name. Use /tools to list available functions."
            )
        resolved[name] = function
    return resolved


def resolve_tool_definitions(surface: ToolSurface, names: Iterable[str]) -> list[SurfaceTool]:
    """Resolve mentioned names to PTC functions, deduplicated in mention order.

    Raises ``ToolMentionError`` naming the first mention that is not a current function.
    """
    functions = _resolve_mentioned_functions(surface, names).values()
    return list({function.python_name: function for function in functions}.values())


def rewrite_tool_mentions(surface: ToolSurface, texts: Sequence[str]) -> ToolMentionRewrite:
    """Resolve ``^tool:`` mentions in ``texts`` and rewrite them for the model.

    A mention inside the prose becomes the function's Python name in backticks. Mentions that
    end a text (attachment-style, as the completion popup appends them) are removed together
    with the whitespace before them. Everything else is preserved byte for byte.

    Raises ``ToolMentionError`` for a mention that is not a current PTC function.
    """
    matches = [list(iter_tool_mentions(text)) for text in texts]
    names = [mention_name(match) for text_matches in matches for match in text_matches]
    if not names:
        return ToolMentionRewrite(texts=tuple(texts), definitions=None)

    functions = _resolve_mentioned_functions(surface, names)
    assert surface.programmatic is not None
    rewritten = tuple(
        _rewrite_text(text, text_matches, functions)
        for text, text_matches in zip(texts, matches, strict=True)
    )
    definitions = render_tool_definitions(
        list({function.python_name: function for function in functions.values()}.values()),
        tool_name=surface.programmatic.tool_name,
    )
    return ToolMentionRewrite(texts=rewritten, definitions=definitions)


def _rewrite_text(
    text: str, matches: Sequence[re.Match[str]], functions: dict[str, SurfaceTool]
) -> str:
    end = len(text.rstrip())
    inline = len(matches)
    while inline and matches[inline - 1].end() == end:
        inline -= 1
        end = len(text[: matches[inline].start()].rstrip())
    if inline == len(matches):
        end = len(text)

    pieces: list[str] = []
    cursor = 0
    for match in matches[:inline]:
        pieces.append(text[cursor : match.start()])
        pieces.append(f"`{functions[mention_name(match)].python_name}`")
        cursor = match.end()
    pieces.append(text[cursor:end])
    return "".join(pieces)


def expand_tool_mentions(surface: ToolSurface, message: PromptMessageExtended) -> None:
    """Rewrite ``^tool:`` mentions in a message's text blocks and append their definitions.

    Does nothing when PTC is off for the agent: then ``^tool:`` is ordinary text. Rewritten
    blocks keep the typed text in ``ORIGINAL_TEXT_META``. Raises ``ToolMentionError``.
    """
    if surface.programmatic is None:
        return
    blocks = [block for block in message.content if isinstance(block, TextContent)]
    rewrite = rewrite_tool_mentions(surface, [block.text for block in blocks])
    if rewrite.definitions is None:
        return
    for block, text in zip(blocks, rewrite.texts, strict=True):
        if text != block.text:
            block.meta = {**(block.meta or {}), ORIGINAL_TEXT_META: block.text}
            block.text = text
    message.content.append(rewrite.definitions)


async def expand_agent_tool_mentions(agent: object, message: PromptMessageExtended) -> None:
    """``expand_tool_mentions`` against ``agent``'s surface, looked up only for mentions."""
    if any(
        tool_mention_names(block.text)
        for block in message.content
        if isinstance(block, TextContent)
    ):
        expand_tool_mentions(await resolve_tool_mention_surface(agent), message)


def render_tool_definitions(
    functions: Sequence[SurfaceTool], *, tool_name: str = "exec_sandbox"
) -> TextContent:
    """Render function stubs as one context block for a user message."""
    stubs = "\n\n".join(function.stub or "" for function in functions)
    text = (
        f"<{TOOL_DEFINITIONS_TAG}>\n"
        f"Selected `{tool_name}` function definitions:\n"
        f"```python\n{stubs}\n```\n"
        f"</{TOOL_DEFINITIONS_TAG}>"
    )
    return TextContent(
        type="text",
        text=text,
        meta={TOOL_DEFINITIONS_META: [function.python_name for function in functions]},
    )
