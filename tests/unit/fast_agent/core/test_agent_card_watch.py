from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from fast_agent import FastAgent

if TYPE_CHECKING:
    from pathlib import Path


def _write_agent_card(
    path: Path,
    *,
    name: str = "watcher",
    function_tools: list[str] | None = None,
) -> None:
    lines = [
        "---",
        "type: agent",
        f"name: {name}",
    ]
    if function_tools:
        lines.append("function_tools:")
        lines.extend([f"  - {spec}" for spec in function_tools])
    lines.extend(
        [
            "---",
            "Return ok.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.mark.asyncio
async def test_reload_agents_detects_function_tool_change(tmp_path: Path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("", encoding="utf-8")

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    tool_path = agents_dir / "tools.py"
    tool_path.write_text("def echo():\n    return 'ok'\n", encoding="utf-8")

    card_path = agents_dir / "watcher.md"
    _write_agent_card(card_path, function_tools=["tools.py:echo"])

    fast = FastAgent(
        "watch-test",
        config_path=str(config_path),
        parse_cli_args=False,
        quiet=True,
    )
    fast.load_agents(agents_dir)

    tool_path.write_text("def echo():\n    return 'changed'\n", encoding="utf-8")

    changed = await fast.reload_agents()

    assert changed is True


@pytest.mark.asyncio
async def test_watch_agent_cards_triggers_reload(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("", encoding="utf-8")

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    card_path = agents_dir / "watcher.md"
    _write_agent_card(card_path)

    fast = FastAgent(
        "watch-test",
        config_path=str(config_path),
        parse_cli_args=False,
        quiet=True,
    )
    fast.load_agents(agents_dir)

    reload_mock = AsyncMock(return_value=True)
    fast._agent_card_watch_reload = reload_mock

    async def fake_awatch(*_paths: Path, **_kwargs):
        yield {("modified", card_path)}

    import watchfiles

    monkeypatch.setattr(watchfiles, "awatch", fake_awatch)

    await fast._watch_agent_cards()

    reload_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_reload_agents_skips_invalid_new_card(tmp_path: Path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("", encoding="utf-8")

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    card_path = agents_dir / "watcher.md"
    _write_agent_card(card_path)

    fast = FastAgent(
        "watch-test",
        config_path=str(config_path),
        parse_cli_args=False,
        quiet=True,
    )
    fast.load_agents(agents_dir)

    invalid_path = agents_dir / "sizer.md"
    # An explicitly empty instruction is invalid (empty file now gets default instruction)
    invalid_path.write_text("---\ninstruction: ''\n---\n", encoding="utf-8")

    await fast.reload_agents()
    assert "sizer" not in fast.agents

    _write_agent_card(invalid_path, name="sizer")
    await fast.reload_agents()
    assert "sizer" in fast.agents
