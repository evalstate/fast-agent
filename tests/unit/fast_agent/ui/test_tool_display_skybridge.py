from pydantic import AnyUrl, TypeAdapter

from fast_agent.mcp.skybridge import (
    AppIntegrationKind,
    SkybridgeResourceConfig,
    SkybridgeServerConfig,
    SkybridgeToolConfig,
)
from fast_agent.ui import console
from fast_agent.ui.console_display import ConsoleDisplay

_ANY_URL = TypeAdapter(AnyUrl)


def test_skybridge_summary_warning_prints_bracketed_text_literally() -> None:
    display = ConsoleDisplay()

    with console.console.capture() as capture:
        display.show_skybridge_summary(
            "agent",
            {
                "demo": SkybridgeServerConfig(
                    server_name="demo",
                    warnings=["[draft] resource unavailable"],
                )
            },
        )

    assert "- demo [draft] resource unavailable" in capture.get()


def test_skybridge_summary_prints_bracketed_names_literally() -> None:
    display = ConsoleDisplay()

    with console.console.capture() as capture:
        display.show_skybridge_summary(
            "agent",
            {
                "demo [draft]": SkybridgeServerConfig(
                    server_name="demo [draft]",
                    tools=[
                        SkybridgeToolConfig(
                            tool_name="[open]",
                            namespaced_tool_name="demo/[open]",
                            kind=AppIntegrationKind.SKYBRIDGE,
                            is_valid=True,
                        )
                    ],
                )
            },
        )

    rendered = capture.get()
    assert "demo [draft]" in rendered
    assert "demo/[open]" in rendered


def test_skybridge_summary_pluralizes_integration_counts() -> None:
    display = ConsoleDisplay()

    with console.console.capture() as capture:
        display.show_skybridge_summary(
            "agent",
            {
                "demo": SkybridgeServerConfig(
                    server_name="demo",
                    ui_resources=[
                        SkybridgeResourceConfig(
                            uri=_ANY_URL.validate_python("https://example.test/app.html"),
                            is_mcp_app=True,
                        )
                    ],
                    tools=[
                        SkybridgeToolConfig(
                            tool_name="open",
                            namespaced_tool_name="demo/open",
                            kind=AppIntegrationKind.SKYBRIDGE,
                            is_valid=True,
                        )
                    ],
                )
            },
        )

    rendered = capture.get()
    assert "MCP Apps: 0 tools, 1 resource" in rendered
    assert "OpenAI Apps SDK: 1 tool, 0 resources" in rendered
