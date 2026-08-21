from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fast_agent.plugins.models import PluginPostUserTurnSpec
from fast_agent.plugins.post_user_turn import (
    load_plugin_post_user_turn_handlers,
    run_plugin_post_user_turn,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_post_user_turn_runs_sync_and_async_handlers_in_order(tmp_path: Path) -> None:
    sync_file = tmp_path / "sync.py"
    sync_file.write_text(
        "def display(ctx):\n    return f\"sync:{ctx.config['label']}\"\n",
        encoding="utf-8",
    )
    async_file = tmp_path / "async.py"
    async_file.write_text(
        "async def display(ctx):\n    return f'async:{ctx.agent_name}'\n",
        encoding="utf-8",
    )
    handlers = load_plugin_post_user_turn_handlers(
        [
            PluginPostUserTurnSpec("sync", f"{sync_file}:display"),
            PluginPostUserTurnSpec("async", f"{async_file}:display"),
        ]
    )
    displayed: list[str] = []

    await run_plugin_post_user_turn(
        handlers,
        agent_name="assistant",
        turn_usage=(),
        session_usage=(),
        config={"sync": {"label": "cost"}},
        display=displayed.append,
    )

    assert displayed == ["sync:cost", "async:assistant"]


@pytest.mark.asyncio
async def test_failing_post_user_turn_plugin_does_not_block_later_plugin(
    tmp_path: Path,
) -> None:
    plugin_file = tmp_path / "hooks.py"
    plugin_file.write_text(
        "def broken(ctx):\n"
        "    raise RuntimeError('broken')\n"
        "\n"
        "def working(ctx):\n"
        "    return 'working'\n",
        encoding="utf-8",
    )
    handlers = load_plugin_post_user_turn_handlers(
        [
            PluginPostUserTurnSpec("broken", f"{plugin_file}:broken"),
            PluginPostUserTurnSpec("working", f"{plugin_file}:working"),
        ]
    )
    displayed: list[str] = []

    await run_plugin_post_user_turn(
        handlers,
        agent_name="assistant",
        turn_usage=(),
        session_usage=(),
        config={},
        display=displayed.append,
    )

    assert displayed == ["working"]


@pytest.mark.asyncio
async def test_structured_post_user_turn_result_reports_session_usage(
    tmp_path: Path,
) -> None:
    plugin_file = tmp_path / "structured.py"
    plugin_file.write_text(
        "from fast_agent.plugins import PluginPostUserTurnOutput\n"
        "\n"
        "def display(ctx):\n"
        "    return PluginPostUserTurnOutput(\n"
        "        display='Cost: $0.01 last',\n"
        "        session_usage='$0.12',\n"
        "    )\n",
        encoding="utf-8",
    )
    handlers = load_plugin_post_user_turn_handlers(
        [PluginPostUserTurnSpec("cost", f"{plugin_file}:display")]
    )
    displayed: list[str] = []
    reported: list[str] = []

    await run_plugin_post_user_turn(
        handlers,
        agent_name="assistant",
        turn_usage=(),
        session_usage=(),
        config={},
        display=displayed.append,
        report_session_usage=reported.append,
    )

    assert displayed == ["Cost: $0.01 last"]
    assert reported == ["$0.12"]
