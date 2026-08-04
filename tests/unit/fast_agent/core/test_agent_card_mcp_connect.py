from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from fast_agent import FastAgent
from fast_agent.agents.agent_types import AgentConfig, MCPConnectTarget
from fast_agent.core.agent_card_runtime import AgentCardRuntimeMixin
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


def _write_provider_mcp_card(path: Path) -> None:
    lines = [
        "---",
        "type: agent",
        "name: card_agent",
        "mcp_connect:",
        "  - target: 'https://mcp.stripe.com'",
        "    name: 'stripe_remote'",
        "    management: provider",
        "    access_token: 'Bearer provider-token'",
        "---",
        "Return ok.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_provider_connector_card(path: Path) -> None:
    lines = [
        "---",
        "type: agent",
        "name: card_agent",
        "mcp_connect:",
        "  - name: 'dropbox'",
        "    management: provider",
        "    connector_id: 'connector_dropbox'",
        "    access_token: 'Bearer provider-token'",
        "    defer_loading: true",
        "---",
        "Return ok.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _card_server(fast: FastAgent, visible_name: str):
    cfg = fast.app.context.config
    assert cfg is not None
    assert cfg.mcp is not None
    matches = [
        (name, server)
        for name, server in cfg.mcp.servers.items()
        if name.startswith("card-") and server.name == visible_name
    ]
    assert len(matches) == 1
    return matches[0]


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
        assert "bar" not in config.servers

        context = fast.app.context
        cfg = context.config
        assert cfg is not None
        assert cfg.mcp is not None
        internal_name, server_cfg = _card_server(fast, "bar")
        assert internal_name in config.servers
        assert server_cfg.command == "npx"
        assert server_cfg.args == ["@foo/bar"]

        registry_cfg = (
            context.server_registry.registry.get(internal_name) if context.server_registry else None
        )
        assert registry_cfg is not None
        assert registry_cfg.command == "npx"
    finally:
        await fast.app.cleanup()


@pytest.mark.asyncio
async def test_sync_named_mcp_connect_materializes_target_and_protocol_mode(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mcp:",
                "  defaults:",
                "    reconnect_on_disconnect: false",
                "    include_instructions: false",
            ]
        ),
        encoding="utf-8",
    )
    card_path = tmp_path / "card_agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: card_agent",
                "mcp_connect:",
                "  docs:",
                "    target: '@foo/bar'",
                "    protocol_mode: modern",
            ]
        ),
        encoding="utf-8",
    )

    fast = FastAgent(
        "mcp-connect-test",
        config_path=str(config_path),
        parse_cli_args=False,
        quiet=True,
    )
    fast.load_agents(card_path)

    await fast.app.initialize()
    try:
        fast._sync_agent_card_mcp_servers()

        cfg = fast.app.context.config
        assert cfg is not None
        assert cfg.mcp is not None
        _internal_name, server_cfg = _card_server(fast, "docs")
        assert server_cfg.command == "npx"
        assert server_cfg.args == ["@foo/bar"]
        assert server_cfg.protocol_mode == "modern"
        assert server_cfg.reconnect_on_disconnect is False
        assert server_cfg.include_instructions is False
        assert server_cfg.env is None
        assert server_cfg.cwd is None
    finally:
        await fast.app.cleanup()


@pytest.mark.asyncio
async def test_unrelated_cards_can_use_same_visible_mcp_namespace(tmp_path: Path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("", encoding="utf-8")
    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    for agent_name, target in (
        ("alpha", "https://alpha.example/mcp"),
        ("beta", "https://beta.example/mcp"),
    ):
        (cards_dir / f"{agent_name}.yaml").write_text(
            "\n".join(
                [
                    f"name: {agent_name}",
                    "mcp_connect:",
                    "  docs:",
                    f"    target: {target}",
                ]
            ),
            encoding="utf-8",
        )

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
        card_servers = {
            name: server for name, server in cfg.mcp.servers.items() if name.startswith("card-")
        }
        assert len(card_servers) == 2
        assert {server.name for server in card_servers.values()} == {"docs"}
        alpha_server = fast.agents["alpha"]["config"].servers
        beta_server = fast.agents["beta"]["config"].servers
        assert len(alpha_server) == len(beta_server) == 1
        assert alpha_server != beta_server
        assert set(alpha_server + beta_server) == set(card_servers)
    finally:
        await fast.app.cleanup()


@pytest.mark.asyncio
async def test_internal_card_name_collision_is_rejected_before_config_publication(
    tmp_path: Path,
) -> None:
    card_path = tmp_path / "card_agent.yaml"
    card_path.write_text(
        "name: card_agent\nmcp_connect:\n  docs:\n    target: '@foo/bar'\n",
        encoding="utf-8",
    )
    entry = MCPConnectTarget(target="@foo/bar", name="docs")
    internal_name = AgentCardRuntimeMixin._card_mcp_server_name(
        AgentConfig(name="card_agent", source_path=card_path),
        entry,
        "docs",
    )
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mcp:",
                "  servers:",
                f"    {internal_name}:",
                "      command: echo",
            ]
        ),
        encoding="utf-8",
    )
    fast = FastAgent(
        "mcp-connect-test",
        config_path=str(config_path),
        parse_cli_args=False,
        quiet=True,
    )
    fast.load_agents(card_path)

    await fast.app.initialize()
    try:
        with pytest.raises(AgentConfigError, match="ownership collision"):
            fast._sync_agent_card_mcp_servers()

        cfg = fast.app.context.config
        assert cfg is not None
        assert cfg.mcp is not None
        assert cfg.mcp.servers[internal_name].command == "echo"
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
        _internal_name, server_cfg = _card_server(fast, "demo_remote")
        assert server_cfg.transport == "http"
        assert server_cfg.headers == {"Authorization": "Bearer token-from-card"}
        assert server_cfg.auth is not None
        assert server_cfg.auth.oauth is False
    finally:
        await fast.app.cleanup()


@pytest.mark.asyncio
async def test_sync_agent_card_provider_mcp_connect_normalizes_provider_target(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("", encoding="utf-8")

    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    _write_provider_mcp_card(cards_dir / "card_agent.md")

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
        _internal_name, server_cfg = _card_server(fast, "stripe_remote")
        assert server_cfg.management == "provider"
        assert server_cfg.url == "https://mcp.stripe.com/mcp"
        assert server_cfg.access_token == "provider-token"
        assert server_cfg.headers is None
    finally:
        await fast.app.cleanup()


@pytest.mark.asyncio
async def test_sync_agent_card_provider_connector_registers_runtime_server(tmp_path: Path) -> None:
    config_path = tmp_path / "fastagent.config.yaml"
    config_path.write_text("", encoding="utf-8")

    cards_dir = tmp_path / "cards"
    cards_dir.mkdir()
    _write_provider_connector_card(cards_dir / "card_agent.md")

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
        _internal_name, server_cfg = _card_server(fast, "dropbox")
        assert server_cfg.management == "provider"
        assert server_cfg.connector_id == "connector_dropbox"
        assert server_cfg.access_token == "provider-token"
        assert server_cfg.defer_loading is True
        assert server_cfg.url is None
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
        with pytest.raises(AgentConfigError, match="ownership collision"):
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
        internal_name, _server_cfg = _card_server(fast, "bar")

        _write_card(card_path, include_mcp_connect=False)
        changed = await fast.reload_agents()
        assert changed is True

        fast._sync_agent_card_mcp_servers()
        cfg = fast.app.context.config
        assert cfg is not None
        assert cfg.mcp is not None
        assert internal_name not in cfg.mcp.servers
    finally:
        await fast.app.cleanup()


@pytest.mark.asyncio
async def test_sync_agent_card_mcp_connect_rejects_central_ownership_collision(
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
        with pytest.raises(AgentConfigError, match="ownership collision"):
            fast._sync_agent_card_mcp_servers()
    finally:
        await fast.app.cleanup()
