"""Reference client demo for draft MCP data-layer sessions.

This demo uses fast-agent's MCPAggregator directly (no LLM key required) and
shows:

1. Automatic `sessions/create` after initialize.
2. Session metadata echo/update on tool calls.
3. Explicit `sessions/delete`.
4. Session-not-found behavior after deletion.
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
from fast_agent.mcp.mcp_agent_client_session import MCPAgentClientSession
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



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="fast-agent MCP draft sessions demo")
    parser.add_argument(
        "--advertise-session-capability",
        action="store_true",
        help=(
            "Advertise client test capability (experimental/sessions) during initialize."
        ),
    )
    return parser.parse_args()


async def _live_session(aggregator: MCPAggregator, server_name: str) -> MCPAgentClientSession:
    manager = aggregator._require_connection_manager()  # noqa: SLF001
    server_conn = await manager.get_server(
        server_name,
        client_session_factory=aggregator._create_session_factory(server_name),  # noqa: SLF001
    )
    session = server_conn.session
    if isinstance(session, MCPAgentClientSession):
        return session
    raise RuntimeError(f"Server '{server_name}' did not return MCPAgentClientSession")


async def main() -> None:
    args = _parse_args()

    server_settings = _server_settings_stdio(
        advertise_session_capability=args.advertise_session_capability
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
            await _print_status(aggregator, "after initialize (auto-create)")

            for note in ("first turn", "second turn"):
                result = await aggregator.call_tool(
                    "session_probe",
                    {"note": note},
                )
                print(f"\n[probe note={note!r}] {_extract_text(result)}")
                await _print_status(aggregator, f"post probe ({note})")

            session = await _live_session(aggregator, SERVER_NAME)
            deleted = await session.experimental_session_delete()
            print(f"\n[sessions/delete] deleted={deleted}")
            await _print_status(aggregator, "after sessions/delete")

            post_delete = await aggregator.call_tool(
                "session_probe",
                {"note": "after-delete"},
            )
            if post_delete.isError:
                print(f"\n[expected error after delete] {_extract_text(post_delete)}")

            created = await session.experimental_session_create()
            print(f"\n[sessions/create] {created}")

            result = await aggregator.call_tool(
                "session_probe",
                {"note": "after-recreate"},
            )
            print(f"\n[probe after recreate] {_extract_text(result)}")
            await _print_status(aggregator, "final")
    finally:
        await asyncio.sleep(0.05)
        await LoggingConfig.shutdown()
        await AsyncEventBus.get().stop()
        AsyncEventBus.reset()
        await asyncio.sleep(0.05)


if __name__ == "__main__":
    asyncio.run(main())
