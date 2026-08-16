#!/usr/bin/env python3
"""MCP app integration test server exposing multiple scenarios."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from fastmcp import FastMCP

from fast_agent.mcp.app_integrations.mcp_apps import MCP_APPS_MIME_TYPE
from fast_agent.mcp.app_integrations.openai_apps_sdk import OPENAI_APPS_SDK_MIME_TYPE

if TYPE_CHECKING:
    from collections.abc import Sequence

    from fastmcp.tools import Tool


class AppIntegrationTestServer(FastMCP):
    """FastMCP server that decorates tool listings with app metadata."""

    def __init__(
        self,
        *args,
        tool_metadata: dict[str, dict[str, object]] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._tool_metadata = tool_metadata or {}

    async def list_tools(self, *, run_middleware: bool = True) -> Sequence[Tool]:
        tools = await super().list_tools(run_middleware=run_middleware)
        for tool in tools:
            metadata = self._tool_metadata.get(tool.name)
            if metadata:
                tool.meta = metadata
        return tools


def build_valid_scenario() -> AppIntegrationTestServer:
    server = AppIntegrationTestServer(
        name="OpenAI Apps SDK Valid Scenario",
        tool_metadata={
            "render_valid_widget": {"openai/outputTemplate": "ui://openai-apps-sdk/widget-valid"}
        },
    )

    @server.tool(
        name="render_valid_widget",
        description="Return HTML for a valid OpenAI Apps SDK widget",
    )
    def render_valid_widget() -> str:
        return "<html><body><h1>Valid OpenAI Apps SDK Widget</h1></body></html>"

    @server.resource(
        "ui://openai-apps-sdk/widget-valid",
        description="Valid OpenAI Apps SDK resource",
        mime_type=OPENAI_APPS_SDK_MIME_TYPE,
    )
    def valid_widget_resource() -> str:
        return "<html><body><h1>Valid OpenAI Apps SDK Widget</h1></body></html>"

    return server


def build_invalid_mime_scenario() -> AppIntegrationTestServer:
    server = AppIntegrationTestServer(
        name="OpenAI Apps SDK Invalid MIME Scenario",
        tool_metadata={
            "render_invalid_widget": {
                "openai/outputTemplate": "ui://openai-apps-sdk/widget-invalid"
            }
        },
    )

    @server.tool(
        name="render_invalid_widget",
        description="Return HTML that lacks the OpenAI Apps SDK MIME type",
    )
    def render_invalid_widget() -> str:
        return "<html><body><h1>Invalid MIME</h1></body></html>"

    @server.resource(
        "ui://openai-apps-sdk/widget-invalid",
        description="Resource served with a non-OpenAI Apps SDK MIME type",
        mime_type="text/html",
    )
    def invalid_widget_resource() -> str:
        return "<html><body><h1>Invalid MIME</h1></body></html>"

    return server


def build_missing_resource_scenario() -> AppIntegrationTestServer:
    server = AppIntegrationTestServer(
        name="OpenAI Apps SDK Missing Resource Scenario",
        tool_metadata={
            "render_missing_widget": {
                "openai/outputTemplate": "ui://openai-apps-sdk/widget-missing"
            }
        },
    )

    @server.tool(
        name="render_missing_widget",
        description="Advertises a template that does not exist on the server",
    )
    def render_missing_widget() -> str:
        return "<html><body><h1>Missing Resource</h1></body></html>"

    @server.resource(
        "ui://openai-apps-sdk/orphan-widget",
        description="Orphaned OpenAI Apps SDK resource with no tool linkage",
        mime_type=OPENAI_APPS_SDK_MIME_TYPE,
    )
    def orphan_widget_resource() -> str:
        return "<html><body><h1>Orphan Widget</h1></body></html>"

    return server


def build_mcp_apps_scenario() -> AppIntegrationTestServer:
    server = AppIntegrationTestServer(
        name="MCP Apps Valid Scenario",
        tool_metadata={
            "open_workspace": {
                "ui": {
                    "resourceUri": "ui://mcp-apps/workspace",
                    "visibility": ["model", "app"],
                }
            }
        },
    )

    @server.tool(
        name="open_workspace",
        description="Open a valid MCP Apps workspace",
    )
    def open_workspace() -> str:
        return "Workspace ready"

    @server.resource(
        "ui://mcp-apps/workspace",
        description="Valid MCP Apps resource",
        mime_type=MCP_APPS_MIME_TYPE,
    )
    def workspace_resource() -> str:
        return "<html><body><h1>MCP Apps Workspace</h1></body></html>"

    return server


SCENARIO_BUILDERS = {
    "valid": build_valid_scenario,
    "invalid-mime": build_invalid_mime_scenario,
    "missing-resource": build_missing_resource_scenario,
    "mcp-apps": build_mcp_apps_scenario,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP app integration test server scenarios")
    parser.add_argument(
        "scenario",
        choices=SCENARIO_BUILDERS.keys(),
        help="Which app integration scenario to run",
    )
    args = parser.parse_args()

    server_factory = SCENARIO_BUILDERS[args.scenario]
    app = server_factory()
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
