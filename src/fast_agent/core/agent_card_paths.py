from __future__ import annotations

from typing import TYPE_CHECKING, Final

from fast_agent.utils.text import strip_casefold

if TYPE_CHECKING:
    from pathlib import Path

MARKDOWN_AGENT_CARD_EXTENSIONS: Final[frozenset[str]] = frozenset((".md", ".markdown"))
YAML_AGENT_CARD_EXTENSIONS: Final[frozenset[str]] = frozenset((".yaml", ".yml"))
AGENT_CARD_EXTENSIONS: Final[frozenset[str]] = (
    MARKDOWN_AGENT_CARD_EXTENSIONS | YAML_AGENT_CARD_EXTENSIONS
)


def agent_card_suffix(path: Path) -> str:
    return strip_casefold(path.suffix)


def is_agent_card_path(path: Path) -> bool:
    return agent_card_suffix(path) in AGENT_CARD_EXTENSIONS


def is_markdown_agent_card_path(path: Path) -> bool:
    return agent_card_suffix(path) in MARKDOWN_AGENT_CARD_EXTENSIONS


def is_yaml_agent_card_path(path: Path) -> bool:
    return agent_card_suffix(path) in YAML_AGENT_CARD_EXTENSIONS
