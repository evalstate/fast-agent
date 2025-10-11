from fast_agent.mcp.skybridge import (
    SkybridgeResourceConfig,
    SkybridgeServerConfig,
    SkybridgeToolConfig,
)
from fast_agent.ui.console_display import ConsoleDisplay


def test_summarize_skybridge_configs_flags_invalid_resource() -> None:
    resource_warning = "served as 'text/html' instead of 'text/html+skybridge'"
    resource = SkybridgeResourceConfig(
        uri="ui://widget/pizza-map.html",
        mime_type="text/html",
        is_skybridge=False,
        warning=resource_warning,
    )
    tool_warning = (
        "Tool 'hf/pizzaz-pizza-map' references resource 'ui://widget/pizza-map.html' "
        "served as 'text/html' instead of 'text/html+skybridge'"
    )
    tool = SkybridgeToolConfig(
        tool_name="pizzaz-pizza-map",
        namespaced_tool_name="hf/pizzaz-pizza-map",
        template_uri="ui://widget/pizza-map.html",
        is_valid=False,
        warning=tool_warning,
    )
    config = SkybridgeServerConfig(
        server_name="hf",
        supports_resources=True,
        ui_resources=[resource],
        warnings=[f"{resource.uri}: {resource_warning}", tool_warning],
        tools=[tool],
    )

    rows, warnings = ConsoleDisplay.summarize_skybridge_configs({"hf": config})

    assert len(rows) == 1
    row = rows[0]
    assert row["server_name"] == "hf"
    assert row["enabled"] is False
    assert row["valid_resource_count"] == 0
    assert row["total_resource_count"] == 1
    assert row["active_tools"] == []

    assert len(warnings) == 2
    assert any("ui://widget/pizza-map.html" in warning for warning in warnings)
    assert any("pizzaz-pizza-map" in warning for warning in warnings)


def test_summarize_skybridge_configs_ignores_servers_without_signals() -> None:
    config = SkybridgeServerConfig(server_name="empty")

    rows, warnings = ConsoleDisplay.summarize_skybridge_configs({"empty": config})

    assert rows == []
    assert warnings == []
