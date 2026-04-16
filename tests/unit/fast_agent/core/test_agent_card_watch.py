from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock

import pytest

from fast_agent import FastAgent
from fast_agent.core.fastagent import AgentRefreshResult
from fast_agent.session import (
    SessionContinuationSnapshot,
    SessionHydrationResult,
    SessionSnapshot,
    get_session_manager,
    reset_session_manager,
)
from fast_agent.types import PromptMessageExtended, text_content

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.interfaces import AgentProtocol


def _write_agent_card(
    path: Path,
    *,
    name: str = "watcher",
    function_tools: list[object] | None = None,
    messages_file: str | None = None,
) -> None:
    lines = [
        "---",
        "type: agent",
        f"name: {name}",
    ]
    if function_tools:
        import yaml

        lines.append("function_tools:")
        for spec in function_tools:
            dumped = yaml.safe_dump(spec, sort_keys=False).rstrip().splitlines()
            if len(dumped) == 1:
                lines.append(f"  - {dumped[0]}")
                continue
            first, *rest = dumped
            lines.append(f"  - {first}")
            lines.extend([f"    {entry}" for entry in rest])
    if messages_file:
        lines.append(f"messages: {messages_file}")
    lines.extend(
        [
            "---",
            "Return ok.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_history_json(path: Path, text: str) -> None:
    payload = {"messages": [{"role": "user", "content": {"type": "text", "text": text}}]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
async def test_reload_agents_detects_structured_function_tool_change(tmp_path: Path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("", encoding="utf-8")

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    tool_path = agents_dir / "tools.py"
    tool_path.write_text("def echo(code: str):\n    return code\n", encoding="utf-8")

    card_path = agents_dir / "watcher.md"
    _write_agent_card(
        card_path,
        function_tools=[
            {
                "entrypoint": "tools.py:echo",
                "variant": "code",
                "language": "python",
            }
        ],
    )

    fast = FastAgent(
        "watch-test",
        config_path=str(config_path),
        parse_cli_args=False,
        quiet=True,
    )
    fast.load_agents(agents_dir)

    tool_path.write_text("def echo(code: str):\n    return code.upper()\n", encoding="utf-8")

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


@pytest.mark.asyncio
async def test_reload_agents_detects_new_card(tmp_path: Path) -> None:
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

    new_card_path = agents_dir / "sizer.md"
    _write_agent_card(new_card_path, name="sizer")

    changed = await fast.reload_agents()

    assert changed is True
    assert "sizer" in fast.agents


@pytest.mark.asyncio
async def test_reload_agents_prunes_removed_child_agents(tmp_path: Path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("", encoding="utf-8")

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    parent_path = agents_dir / "parent.md"
    child_path = agents_dir / "child.md"
    _write_agent_card(parent_path, name="parent")
    _write_agent_card(child_path, name="child")

    fast = FastAgent(
        "watch-test",
        config_path=str(config_path),
        parse_cli_args=False,
        quiet=True,
    )
    fast.load_agents(agents_dir)
    fast.attach_agent_tools("parent", ["child"])

    child_path.unlink()
    changed = await fast.reload_agents()

    assert changed is True
    assert "child" not in fast.agents
    parent_data = fast.agents["parent"]
    assert "child" not in (parent_data.get("child_agents") or [])


@pytest.mark.asyncio
async def test_reload_agents_discards_unsaved_live_history_without_session(
    monkeypatch,
    tmp_path: Path,
) -> None:
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
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    fast.args.watch = True
    fast.load_agents(agents_dir)

    async with fast.run() as app:
        agent = app["watcher"]
        history_message = PromptMessageExtended(
            role="user",
            content=[text_content("hello")],
        )
        agent.message_history.append(history_message)

        card_path.write_text(
            "---\ntype: agent\nname: watcher\n---\nReturn ok updated.\n",
            encoding="utf-8",
        )
        changed = await fast.reload_agents()
        assert changed is True

        await app.refresh_if_needed()
        updated_agent = app["watcher"]
        assert updated_agent.message_history == []


@pytest.mark.asyncio
async def test_reload_agents_updates_history_when_file_newer(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("", encoding="utf-8")

    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    history_path = agents_dir / "history.json"
    _write_history_json(history_path, "first")
    card_path = agents_dir / "watcher.md"
    _write_agent_card(card_path, messages_file="history.json")

    fast = FastAgent(
        "watch-test",
        config_path=str(config_path),
        parse_cli_args=False,
        quiet=True,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    fast.args.watch = True
    fast.load_agents(agents_dir)

    async with fast.run() as app:
        agent = app["watcher"]
        assert agent.message_history
        assert agent.message_history[0].all_text() == "first"

        _write_history_json(history_path, "second")
        new_ts = time.time() + 2.0
        os.utime(history_path, (new_ts, new_ts))

        changed = await fast.reload_agents()
        assert changed is True

        await app.refresh_if_needed()
        updated_agent = app["watcher"]
        assert updated_agent.message_history
        assert updated_agent.message_history[0].all_text() == "second"


@pytest.mark.asyncio
async def test_reload_agents_rehydrates_saved_session_not_unsaved_live_history(
    monkeypatch,
    tmp_path: Path,
) -> None:
    reset_session_manager()
    monkeypatch.chdir(tmp_path)

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
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    fast.args.watch = True
    fast.load_agents(agents_dir)

    try:
        async with fast.run() as app:
            agent = app["watcher"]
            saved_message = PromptMessageExtended(
                role="user",
                content=[text_content("saved turn")],
            )
            unsaved_message = PromptMessageExtended(
                role="assistant",
                content=[text_content("unsaved drift")],
            )
            agent.message_history.append(saved_message)

            manager = get_session_manager()
            await manager.save_current_session(agent, agent_registry=app.registered_agents())

            agent.message_history.append(unsaved_message)

            card_path.write_text(
                "---\ntype: agent\nname: watcher\n---\nReturn ok updated.\n",
                encoding="utf-8",
            )

            changed = await fast.reload_agents()
            assert changed is True

            await app.refresh_if_needed()
            updated_agent = app["watcher"]
            assert [message.all_text() for message in updated_agent.message_history] == ["saved turn"]
            assert updated_agent.instruction.strip() == "Return ok updated."
    finally:
        reset_session_manager()


@pytest.mark.asyncio
async def test_refresh_result_rehydrates_only_updated_agents_on_partial_refresh(tmp_path: Path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("", encoding="utf-8")
    fast = FastAgent(
        "watch-test",
        config_path=str(config_path),
        parse_cli_args=False,
        quiet=True,
    )

    rehydrated_agent_sets: list[set[str]] = []

    class _Agent:
        def __init__(self, name: str) -> None:
            self.name = name

    def fake_hydrate_active_agents_from_session(agents):
        rehydrated_agent_sets.append(set(agents))
        now = datetime.now()
        return SessionHydrationResult(
            session=None,  # type: ignore[arg-type]
            snapshot=SessionSnapshot(
                session_id="s-1",
                created_at=now,
                last_activity=now,
                continuation=SessionContinuationSnapshot(),
            ),
            loaded_agents={},
            restored_prompts={},
            skipped_agents=[],
            missing_history_files=[],
            active_agent="unchanged",
        )

    fast._hydrate_active_agents_from_session = fake_hydrate_active_agents_from_session  # type: ignore[method-assign]

    changed_agent = cast("AgentProtocol", _Agent("changed"))
    unchanged_agent = cast("AgentProtocol", _Agent("unchanged"))

    result = await fast._refresh_result_from_session_restore(
        {"changed": changed_agent, "unchanged": unchanged_agent},
        {"changed": changed_agent},
    )

    assert isinstance(result, AgentRefreshResult)
    assert rehydrated_agent_sets == [{"changed"}]
    assert result.active_agent is None
