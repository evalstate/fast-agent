"""Run an end-to-end fast-agent demo against the experimental session server.

This script does not require an LLM API key. It talks to MCP directly via
``MCPAggregator`` and prints status snapshots using the same renderer used by
``/mcp`` in interactive mode.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

from mcp.types import CallToolResult, TextContent

from fast_agent.config import MCPServerSettings
from fast_agent.context import Context
from fast_agent.core.logging.logger import LoggingConfig
from fast_agent.core.logging.transport import AsyncEventBus
from fast_agent.mcp.mcp_aggregator import MCPAggregator
from fast_agent.mcp_server_registry import ServerRegistry
from fast_agent.ui.mcp_display import render_mcp_status

SERVER_NAME = "experimental-sessions"


class _StatusAdapter:
    """Small adapter so we can reuse ``render_mcp_status`` directly."""

    def __init__(self, aggregator: MCPAggregator) -> None:
        self._aggregator = aggregator
        self.config = SimpleNamespace(instruction="")

    async def get_server_status(self):
        return await self._aggregator.collect_server_status()


def _extract_text(result: CallToolResult) -> str:
    parts = [
        item.text
        for item in result.content
        if isinstance(item, TextContent) and item.text.strip()
    ]
    return "\n".join(parts) if parts else "<no text content>"


async def _print_status(aggregator: MCPAggregator, label: str) -> None:
    print(f"\n=== {label} ===")
    await render_mcp_status(_StatusAdapter(aggregator), indent="  ")


def _server_settings() -> MCPServerSettings:
    return _server_settings_stdio(advertise_session_capability=False)


def _server_settings_stdio(*, advertise_session_capability: bool) -> MCPServerSettings:
    repo_root = Path(__file__).resolve().parents[3]
    server_script = repo_root / "examples" / "experimental" / "mcp_sessions" / "session_server.py"
    return MCPServerSettings(
        name=SERVER_NAME,
        transport="stdio",
        command=sys.executable,
        args=[str(server_script)],
        cwd=str(repo_root),
        experimental_session_advertise=advertise_session_capability,
    )


def _server_settings_http(url: str, *, advertise_session_capability: bool) -> MCPServerSettings:
    return MCPServerSettings(
        name=SERVER_NAME,
        transport="http",
        url=url,
        experimental_session_advertise=advertise_session_capability,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="fast-agent MCP experimental sessions demo")
    parser.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default="stdio",
        help="MCP server transport used by the demo client",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8765/mcp",
        help="Streamable HTTP server URL used when --transport=http",
    )
    parser.add_argument(
        "--advertise-session-capability",
        action="store_true",
        help=(
            "Advertise experimental session capability from the client in initialize, "
            "so you can compare server-first vs client-first style negotiation."
        ),
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()

    server_settings = (
        _server_settings_stdio(advertise_session_capability=args.advertise_session_capability)
        if args.transport == "stdio"
        else _server_settings_http(
            args.url,
            advertise_session_capability=args.advertise_session_capability,
        )
    )

    registry = ServerRegistry()
    registry.registry = {SERVER_NAME: server_settings}

    context = Context(server_registry=registry)
    aggregator = MCPAggregator(
        server_names=[SERVER_NAME],
        connection_persistence=True,
        context=context,
        name="sessions-demo-agent",
    )

    try:
        async with aggregator:
            await _print_status(aggregator, "after initialize")

            steps = [
                ("status", "first turn"),
                ("status", "second turn"),
                ("revoke", "clear cookie"),
                ("status", "after revoke"),
                ("new", "rotate session"),
            ]

            for action, note in steps:
                result = await aggregator.call_tool(
                    "session_probe",
                    {"action": action, "note": note},
                )
                print(f"\n[action={action}] {_extract_text(result)}")
                await _print_status(aggregator, f"post {action}")
    finally:
        # Allow any in-flight logging emissions scheduled by teardown paths to drain.
        await asyncio.sleep(0.05)
        await LoggingConfig.shutdown()
        await AsyncEventBus.get().stop()
        AsyncEventBus.reset()
        await asyncio.sleep(0.05)


if __name__ == "__main__":
    asyncio.run(main())
