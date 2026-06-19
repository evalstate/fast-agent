"""Integration tests for ACP /plugins manager commands."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from fast_agent.acp.slash_commands import SlashCommandHandler
from fast_agent.commands.context import StaticAgentProvider
from fast_agent.config import get_settings, update_global_settings

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class StubAgent:
    name: str
    config: Any = field(default_factory=lambda: SimpleNamespace(model=None))


class _StubAppProvider(StaticAgentProvider):
    pass


@dataclass
class StubAgentInstance:
    agents: dict[str, Any] = field(default_factory=dict)
    app: Any = None

    def __post_init__(self) -> None:
        if self.app is None:
            self.app = _StubAppProvider(self.agents)


def _handler(instance: StubAgentInstance, agent_name: str) -> SlashCommandHandler:
    return SlashCommandHandler("test-session", cast("Any", instance), agent_name)


def _write_plugin(plugins_root: Path, name: str) -> None:
    plugin_root = plugins_root / name
    plugin_root.mkdir(parents=True, exist_ok=True)
    (plugin_root / "plugin.yaml").write_text(
        "schema_version: 1\n"
        f"name: {name}\n"
        "description: Test plugin\n"
        "commands:\n"
        f"  {name}:\n"
        "    description: Run test plugin\n"
        "    handler: ./commands.py:run\n",
        encoding="utf-8",
    )
    (plugin_root / "commands.py").write_text(
        "async def run(ctx):\n    return 'ok'\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_acp_plugins_lists_project_and_global_plugins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    global_home = tmp_path / "global-home"
    env_root = tmp_path / "project-env"
    _write_plugin(global_home / "plugins", "discover")
    _write_plugin(env_root / "plugins", "images")
    (global_home / "fast-agent.yaml").write_text(
        "plugins:\n  enabled: ['discover']\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "fast-agent.yaml"
    config_path.write_text(
        f"default_model: passthrough\nenvironment_dir: '{env_root.as_posix()}'\n"
        "plugins:\n  enabled: ['images']\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FAST_AGENT_HOME", global_home.as_posix())

    old_settings = get_settings()
    get_settings(config_path=str(config_path))
    try:
        agent = StubAgent("dev")
        handler = _handler(StubAgentInstance(agents={"dev": agent}), "dev")

        response = await handler.execute_command("plugins", "")

        assert "project plugins directory" in response
        assert "global plugins directory" in response
        assert "images" in response
        assert "discover" in response
    finally:
        update_global_settings(old_settings)
