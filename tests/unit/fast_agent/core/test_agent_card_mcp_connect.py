from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fast_agent import FastAgent
from fast_agent.core.exceptions import AgentConfigError

if TYPE_CHECKING:
    from pathlib import Path


def _write_card(path: Path, *, include_mcp_connect: bool) -> None:
    lines = [
        "---",
        "type: agent",
        "name: card_agent",
    ]
    lines.extend(
        [
            "servers:",
            "  - bar",
        ]
    )
    if include_mcp_connect:
        lines.extend(
            [
                "mcp_connect:",
                "  - target: '@foo/bar'",
            ]
        )
    lines.extend(
        [
            "---",
            "Return ok.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_http_card_with_auth(path: Path) -> None:
    lines = [
        "---",
        "type: agent",
        "name: card_agent",
        "mcp_connect:",
        "  - target: 'https://demo.hf.space'",
        "    name: 'demo_remote'",
        "    headers:",
        "      Authorization: 'Bearer token-from-card'",
        "    auth:",
        "      oauth: false",
        "---",
        "Return ok.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.mark.asyncio
async def test_sync_agent_card_mcp_connect_registers_runtime_server(tmp_path: Path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("", encoding="utf-8")

    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    _write_card(cards_dir / "card_agent.md", include_mcp_connect=True)

    fast = FastAgent(
        "mcp-connect-test",
        config_path=str(config_path),
        parse_cli_args=False,
        quiet=True,
    )
    fast.load_agents(cards_dir)

    await fast.app.initialize()
    try:
        fast._sync_agent_card_mcp_servers()

        config = fast.agents["card_agent"]["config"]
        assert "bar" in config.servers

        context = fast.app.context
        cfg = context.config
        assert cfg is not None
        assert cfg.mcp is not None
        server_cfg = cfg.mcp.servers.get("bar")
        assert server_cfg is not None
        assert server_cfg.command == "npx"
        assert server_cfg.args == ["@foo/bar"]

        registry_cfg = context.server_registry.registry.get("bar") if context.server_registry else None
        assert registry_cfg is not None
        assert registry_cfg.command == "npx"
    finally:
        await fast.app.cleanup()


@pytest.mark.asyncio
async def test_sync_agent_card_mcp_connect_applies_auth_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("", encoding="utf-8")

    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    _write_http_card_with_auth(cards_dir / "card_agent.md")

    fast = FastAgent(
        "mcp-connect-test",
        config_path=str(config_path),
        parse_cli_args=False,
        quiet=True,
    )
    fast.load_agents(cards_dir)

    await fast.app.initialize()
    try:
        fast._sync_agent_card_mcp_servers()

        context = fast.app.context
        cfg = context.config
        assert cfg is not None
        assert cfg.mcp is not None
        server_cfg = cfg.mcp.servers.get("demo_remote")
        assert server_cfg is not None
        assert server_cfg.transport == "http"
        assert server_cfg.headers == {"Authorization": "Bearer token-from-card"}
        assert server_cfg.auth is not None
        assert server_cfg.auth.oauth is False
    finally:
        await fast.app.cleanup()


@pytest.mark.asyncio
async def test_sync_agent_card_mcp_connect_detects_name_collision(tmp_path: Path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mcp:",
                "  servers:",
                "    bar:",
                "      command: uvx",
                "      args:",
                "        - some-other-server",
            ]
        ),
        encoding="utf-8",
    )

    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    _write_card(cards_dir / "card_agent.md", include_mcp_connect=True)

    fast = FastAgent(
        "mcp-connect-test",
        config_path=str(config_path),
        parse_cli_args=False,
        quiet=True,
    )
    fast.load_agents(cards_dir)

    await fast.app.initialize()
    try:
        with pytest.raises(AgentConfigError, match="Server name collision"):
            fast._sync_agent_card_mcp_servers()
    finally:
        await fast.app.cleanup()


@pytest.mark.asyncio
async def test_sync_agent_card_mcp_connect_prunes_removed_runtime_servers(tmp_path: Path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("", encoding="utf-8")

    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    card_path = cards_dir / "card_agent.md"
    _write_card(card_path, include_mcp_connect=True)

    fast = FastAgent(
        "mcp-connect-test",
        config_path=str(config_path),
        parse_cli_args=False,
        quiet=True,
    )
    fast.load_agents(cards_dir)

    await fast.app.initialize()
    try:
        fast._sync_agent_card_mcp_servers()
        cfg = fast.app.context.config
        assert cfg is not None
        assert cfg.mcp is not None
        assert "bar" in cfg.mcp.servers

        _write_card(card_path, include_mcp_connect=False)
        changed = await fast.reload_agents()
        assert changed is True

        fast._sync_agent_card_mcp_servers()
        cfg = fast.app.context.config
        assert cfg is not None
        assert cfg.mcp is not None
        assert "bar" not in cfg.mcp.servers
    finally:
        await fast.app.cleanup()


@pytest.mark.asyncio
async def test_sync_agent_card_mcp_connect_preserves_declared_servers_on_reload(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mcp:",
                "  servers:",
                "    bar:",
                "      command: npx",
                "      args:",
                "        - '@foo/bar'",
            ]
        ),
        encoding="utf-8",
    )

    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    card_path = cards_dir / "card_agent.md"
    _write_card(card_path, include_mcp_connect=True)

    fast = FastAgent(
        "mcp-connect-test",
        config_path=str(config_path),
        parse_cli_args=False,
        quiet=True,
    )
    fast.load_agents(cards_dir)

    await fast.app.initialize()
    try:
        fast._sync_agent_card_mcp_servers()
        assert "bar" in fast.agents["card_agent"]["config"].servers

        _write_card(card_path, include_mcp_connect=False)
        changed = await fast.reload_agents()
        assert changed is True

        fast._sync_agent_card_mcp_servers()
        assert "bar" in fast.agents["card_agent"]["config"].servers
    finally:
        await fast.app.cleanup()
