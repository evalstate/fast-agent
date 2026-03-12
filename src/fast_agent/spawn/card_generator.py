"""Agent card generator — creates markdown agent card files for runtime loading.

Agent cards use YAML frontmatter format compatible with fast-agent's
``load_agent_cards()`` parser. The body text becomes additional instruction.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _sanitize_name(name: str) -> str:
    """Sanitize agent name to a valid filename and identifier."""
    return name.strip().replace(" ", "_").replace("-", "_").lower()


def generate_agent_card(
    name: str,
    instruction: str,
    agent_cards_dir: str | Path,
    available_servers: list[str] | None = None,
    servers: list[str] | None = None,
    model: str | None = None,
    use_history: bool = True,
    extra_instruction: str = "",
) -> Path:
    """Generate an agent card markdown file.

    Args:
        name: Agent name (will be sanitized).
        instruction: Base instruction for the agent.
        agent_cards_dir: Directory to write cards into.
        available_servers: List of valid server names for validation.
        servers: MCP server names to attach.
        model: Override model for this agent.
        use_history: Whether this agent maintains conversation history.
        extra_instruction: Additional instruction text placed in the body.

    Returns:
        Path to the created agent card file.

    Raises:
        ValueError: If name is empty or servers reference unknown MCP servers.
    """
    if not name or not name.strip():
        raise ValueError("Agent name cannot be empty")

    if not instruction or not instruction.strip():
        raise ValueError("Agent instruction cannot be empty")

    safe_name = _sanitize_name(name)
    servers = servers or []
    available = available_servers or []

    # Validate server references
    if available:
        unknown = [s for s in servers if s not in available]
        if unknown:
            raise ValueError(f"Unknown MCP servers: {unknown}. Available: {available}")

    # Build YAML frontmatter
    lines = ["---"]
    lines.append(f"name: {safe_name}")
    lines.append("type: agent")
    lines.append(f"instruction: |\n  {instruction}")

    if servers:
        lines.append("servers:")
        for srv in servers:
            lines.append(f"  - {srv}")

    if model:
        lines.append(f"model: {model}")

    lines.append(f"use_history: {str(use_history).lower()}")
    lines.append("---")

    # Body = extra instruction
    if extra_instruction:
        lines.append("")
        lines.append(extra_instruction)

    lines.append("")

    # Write file
    cards_dir = Path(agent_cards_dir)
    cards_dir.mkdir(parents=True, exist_ok=True)
    card_path = cards_dir / f"{safe_name}.md"
    card_path.write_text("\n".join(lines), encoding="utf-8")

    return card_path


def list_agent_cards(agent_cards_dir: str | Path) -> list[dict[str, str]]:
    """List all agent cards in the given directory.

    Returns:
        List of dicts with 'name' and 'file' keys.
    """
    cards_dir = Path(agent_cards_dir)
    cards_dir.mkdir(parents=True, exist_ok=True)
    return [{"name": f.stem, "file": str(f)} for f in sorted(cards_dir.glob("*.md"))]


def remove_agent_card(name: str, agent_cards_dir: str | Path) -> bool:
    """Remove an agent card file.

    Returns:
        True if file was deleted, False if not found.
    """
    safe_name = _sanitize_name(name)
    card_path = Path(agent_cards_dir) / f"{safe_name}.md"
    if card_path.exists():
        card_path.unlink()
        return True
    return False


def get_agent_card_content(name: str, agent_cards_dir: str | Path) -> str | None:
    """Read the content of an agent card.

    Returns:
        File content as string, or None if not found.
    """
    safe_name = _sanitize_name(name)
    card_path = Path(agent_cards_dir) / f"{safe_name}.md"
    if card_path.exists():
        return card_path.read_text(encoding="utf-8")
    return None
