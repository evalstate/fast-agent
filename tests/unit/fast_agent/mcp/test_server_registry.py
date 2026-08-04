import pytest
from mcp_types import ServerCapabilities

from fast_agent.config import MCPServerSettings, MCPSettings, Settings
from fast_agent.mcp_server_registry import ServerRegistry


def test_default_server_registry_state_is_per_instance() -> None:
    first = ServerRegistry()
    second = ServerRegistry()

    first.registry["demo"] = MCPServerSettings(name="demo", transport="stdio", command="echo")

    assert second.registry == {}


def test_loaded_settings_are_deep_copied_and_tracked_as_central() -> None:
    loaded = MCPServerSettings(
        name="demo",
        transport="stdio",
        command="echo",
        args=["original"],
        env={"TOKEN": "original"},
    )
    settings = Settings.model_construct(mcp=MCPSettings.model_construct(servers={"demo": loaded}))

    registry = ServerRegistry(settings)
    registered = registry.get_server_config("demo")

    assert registered is not None
    assert registered is not loaded
    assert registered.args is not loaded.args
    assert registered.env is not loaded.env
    assert registry.get_server_origin("demo") == "central"

    assert registered.args is not None
    assert registered.env is not None
    registered.args.append("registry")
    registered.env["TOKEN"] = "registry"
    assert loaded.args == ["original"]
    assert loaded.env == {"TOKEN": "original"}


@pytest.mark.parametrize("origin", ["central", "card"])
def test_runtime_registration_rejects_configured_collision(origin: str) -> None:
    registry = ServerRegistry()
    config = MCPServerSettings(name="demo", transport="stdio", command="echo")
    if origin == "central":
        registry.register_central("demo", config)
    else:
        registry.register_card("demo", config)

    with pytest.raises(ValueError, match=f"collides with {origin}"):
        registry.register_runtime("demo", config)


def test_runtime_removal_and_capability_clear_are_explicit() -> None:
    registry = ServerRegistry()
    config = MCPServerSettings(name="demo", transport="stdio", command="echo")
    capabilities = ServerCapabilities.model_validate({"tools": {}})
    registry.register_runtime("demo", config)
    registry.set_server_capabilities("demo", capabilities)

    registry.clear_server_capabilities("demo")
    assert registry.get_server_capabilities("demo") is None
    assert registry.remove_runtime("demo") is True
    assert registry.get_server_config("demo") is None
    assert registry.get_server_origin("demo") is None


def test_runtime_definition_is_removed_after_final_owner() -> None:
    registry = ServerRegistry()
    config = MCPServerSettings(name="demo", transport="stdio", command="echo")
    registry.register_runtime("demo", config, owner="first")
    registry.register_runtime("demo", config, owner="second")

    assert registry.remove_runtime("demo", owner="first") is False
    assert registry.get_runtime_owners("demo") == frozenset({"second"})
    assert registry.get_server_config("demo") == config

    assert registry.remove_runtime("demo", owner="second") is True
    assert registry.get_server_config("demo") is None


def test_runtime_batch_registration_is_atomic_on_configured_collision() -> None:
    registry = ServerRegistry()
    registry.register_central(
        "central",
        MCPServerSettings(name="central", transport="stdio", command="central"),
    )

    with pytest.raises(ValueError, match="collides with central"):
        registry.register_runtime_batch(
            {
                "runtime": MCPServerSettings(
                    name="runtime",
                    transport="stdio",
                    command="runtime",
                ),
                "central": MCPServerSettings(
                    name="central",
                    transport="stdio",
                    command="replacement",
                ),
            },
            owner="cli-startup",
        )

    assert registry.get_server_config("runtime") is None
    assert registry.get_runtime_owners("runtime") == frozenset()
