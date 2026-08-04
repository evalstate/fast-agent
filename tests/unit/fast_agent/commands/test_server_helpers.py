"""Tests for server helper functions."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from fast_agent.cli.commands.server_helpers import generate_server_name


class TestGenerateServerName:
    """Test cases for generate_server_name function."""

    def test_npm_package_with_org(self):
        """Test npm package names with organization prefix."""
        assert (
            generate_server_name("@modelcontextprotocol/server-filesystem") == "server_filesystem"
        )
        assert generate_server_name("@npmorg/mcp-server") == "mcp_server"
        assert generate_server_name("@my-org/my-mcp-server") == "my_mcp_server"

    def test_simple_package_names(self):
        """Test simple package names without org prefix."""
        assert generate_server_name("my-mcp-server") == "my_mcp_server"
        assert generate_server_name("server") == "server"
        assert generate_server_name("mcp_server") == "mcp_server"

    def test_file_paths(self):
        """Test file paths with extensions."""
        assert generate_server_name("./src/my-server.py") == "src_my_server"
        assert generate_server_name("server.py") == "server"
        assert generate_server_name("./mcp-server.js") == "mcp_server"
        assert generate_server_name("app/server.ts") == "app_server"

    def test_special_characters(self):
        """Test handling of special characters."""
        assert generate_server_name("my.server.name") == "my_server_name"
        assert generate_server_name("server-with-dashes") == "server_with_dashes"
        assert generate_server_name("server/with/slashes") == "slashes"
        assert generate_server_name("server@host") == "server_host"

    def test_multiple_underscores(self):
        """Test cleanup of multiple underscores."""
        assert generate_server_name("server--name") == "server_name"
        assert generate_server_name("my___server") == "my_server"

    def test_edge_cases(self):
        """Test edge cases."""
        assert generate_server_name("") == ""
        assert generate_server_name("@") == ""
        assert generate_server_name("./") == ""
        assert generate_server_name("---") == ""
        assert generate_server_name("123-server") == "123_server"

    def test_leading_trailing_cleanup(self):
        """Test removal of leading/trailing underscores."""
        assert generate_server_name("-server-") == "server"
        assert generate_server_name("_server_") == "server"
        assert generate_server_name("@-server-@") == "server"


@pytest.mark.asyncio
async def test_add_servers_to_config_keeps_url_server_auth_block() -> None:
    from fast_agent.cli.commands.server_helpers import add_servers_to_config

    class _FakeApp:
        def __init__(self) -> None:
            registry: dict[str, object] = {}

            def register_runtime(
                server_name: str,
                server: object,
                *,
                owner: str,
            ) -> None:
                assert owner == "cli"
                registry[server_name] = server

            self.context = SimpleNamespace(
                config=SimpleNamespace(),
                server_registry=SimpleNamespace(
                    registry=registry,
                    register_runtime=register_runtime,
                ),
            )

        async def initialize(self) -> None:
            return None

    fast_app = SimpleNamespace(app=_FakeApp())
    await add_servers_to_config(
        fast_app,
        {
            "example": {
                "transport": "http",
                "url": "https://example.com/mcp",
                "auth": {
                    "oauth": True,
                    "client_metadata_url": "https://example.com/oauth/client-metadata.json",
                },
            }
        },
    )

    config = fast_app.app.context.config.mcp.servers["example"]
    assert config.auth is not None
    assert config.auth.client_metadata_url == "https://example.com/oauth/client-metadata.json"


@pytest.mark.asyncio
async def test_register_runtime_servers_does_not_mutate_loaded_settings() -> None:
    from fast_agent.cli.commands.server_helpers import register_runtime_servers
    from fast_agent.config import MCPServerSettings, MCPSettings, Settings
    from fast_agent.mcp_server_registry import ServerRegistry

    settings = Settings.model_construct(
        mcp=MCPSettings.model_construct(servers={}),
    )

    class _App:
        def __init__(self) -> None:
            self.context = SimpleNamespace(
                config=settings,
                server_registry=ServerRegistry(settings),
            )

        async def initialize(self) -> None:
            return None

    fast_app = SimpleNamespace(app=_App())
    await register_runtime_servers(
        fast_app,
        {
            "runtime": MCPServerSettings(
                name="runtime",
                transport="stdio",
                command="echo",
            )
        },
        owner="cli-startup",
    )

    assert settings.mcp is not None
    assert settings.mcp.servers == {}
    assert fast_app.app.context.server_registry.get_server_origin("runtime") == "runtime"


@pytest.mark.asyncio
async def test_cli_startup_rollback_removes_registration_and_cleans_app() -> None:
    from fast_agent.cli.runtime.agent_setup import _rollback_cli_startup
    from fast_agent.config import MCPServerSettings
    from fast_agent.mcp_server_registry import ServerRegistry

    registry = ServerRegistry()
    server = MCPServerSettings(name="runtime", transport="stdio", command="echo")
    registry.register_runtime_batch({"runtime": server}, owner="cli-startup")

    class _App:
        def __init__(self) -> None:
            self.context = SimpleNamespace(server_registry=registry)
            self.cleaned = False

        async def cleanup(self) -> None:
            self.cleaned = True

    app = _App()
    request = SimpleNamespace(
        mode="interactive",
        startup_mcp_servers={"runtime": server},
    )

    await _rollback_cli_startup(
        SimpleNamespace(app=app),
        cast("Any", request),
    )

    assert registry.registry == {}
    assert app.cleaned is True


@pytest.mark.asyncio
async def test_cli_startup_rollback_handles_uninitialized_core() -> None:
    from fast_agent.cli.runtime.agent_setup import _rollback_cli_startup

    class _App:
        cleaned = False

        @property
        def context(self):
            raise RuntimeError("Core not initialized")

        async def cleanup(self) -> None:
            self.cleaned = True

    app = _App()
    request = SimpleNamespace(mode="serve", startup_mcp_servers=None)

    await _rollback_cli_startup(SimpleNamespace(app=app), cast("Any", request))

    assert app.cleaned is True
