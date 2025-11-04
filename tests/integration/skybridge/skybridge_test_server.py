#!/usr/bin/env python3
"""Skybridge-focused MCP test server exposing multiple scenarios."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool

SKYBRIDGE_MIME_TYPE = "text/html+skybridge"


class SkybridgeTestServer(FastMCP):
    """FastMCP server that decorates tool listings with Skybridge meta tags."""

    def __init__(self, *args, tool_templates: dict[str, str] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._tool_templates = tool_templates or {}

    async def list_tools(self) -> list[MCPTool]:
        tools = await super().list_tools()
        for tool in tools:
            template = self._tool_templates.get(tool.name)
            if template:
                tool.meta = {"openai/outputTemplate": template}
        return tools


def build_valid_scenario() -> SkybridgeTestServer:
    server = SkybridgeTestServer(
        name="Skybridge Valid Scenario",
        tool_templates={"render_valid_widget": "ui://skybridge/widget-valid"},
    )

    @server.tool(name="render_valid_widget", description="Return HTML for a valid Skybridge widget")
    def render_valid_widget() -> str:
        return "<html><body><h1>Valid Skybridge Widget</h1></body></html>"

    @server.resource(
        "ui://skybridge/widget-valid",
        description="Valid Skybridge UI resource",
        mime_type=SKYBRIDGE_MIME_TYPE,
    )
    def valid_widget_resource() -> str:
        return "<html><body><h1>Valid Skybridge Widget</h1></body></html>"

    return server


def build_invalid_mime_scenario() -> SkybridgeTestServer:
    server = SkybridgeTestServer(
        name="Skybridge Invalid MIME Scenario",
        tool_templates={"render_invalid_widget": "ui://skybridge/widget-invalid"},
    )

    @server.tool(
        name="render_invalid_widget", description="Return HTML that lacks the Skybridge MIME type"
    )
    def render_invalid_widget() -> str:
        return "<html><body><h1>Invalid MIME</h1></body></html>"

    @server.resource(
        "ui://skybridge/widget-invalid",
        description="Resource served with a non-Skybridge mime type",
        mime_type="text/html",
    )
    def invalid_widget_resource() -> str:
        return "<html><body><h1>Invalid MIME</h1></body></html>"

    return server


def build_missing_resource_scenario() -> SkybridgeTestServer:
    server = SkybridgeTestServer(
        name="Skybridge Missing Resource Scenario",
        tool_templates={"render_missing_widget": "ui://skybridge/widget-missing"},
    )

    @server.tool(
        name="render_missing_widget",
        description="Advertises a template that does not exist on the server",
    )
    def render_missing_widget() -> str:
        return "<html><body><h1>Missing Resource</h1></body></html>"

    @server.resource(
        "ui://skybridge/orphan-widget",
        description="Orphaned Skybridge resource with no tool linkage",
        mime_type=SKYBRIDGE_MIME_TYPE,
    )
    def orphan_widget_resource() -> str:
        return "<html><body><h1>Orphan Widget</h1></body></html>"

    return server


SCENARIO_BUILDERS = {
    "valid": build_valid_scenario,
    "invalid-mime": build_invalid_mime_scenario,
    "missing-resource": build_missing_resource_scenario,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Skybridge MCP test server scenarios")
    parser.add_argument(
        "scenario",
        choices=SCENARIO_BUILDERS.keys(),
        help="Which Skybridge scenario to run",
    )
    args = parser.parse_args()

    server_factory = SCENARIO_BUILDERS[args.scenario]
    app = server_factory()
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
