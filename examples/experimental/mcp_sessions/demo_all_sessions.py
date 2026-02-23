"""End-to-end demo of all experimental session servers.

No LLM API key required. Runs each server demo as a separate subprocess
to avoid Python 3.13/uvloop child-watcher limitations with multiple
simultaneous stdio servers.
"""

from __future__ import annotations

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

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parents[2]


def _settings(name: str, script: str) -> MCPServerSettings:
    return MCPServerSettings(
        name=name,
        transport="stdio",
        command=sys.executable,
        args=[str(SCRIPTS_DIR / script)],
        cwd=str(REPO_ROOT),
    )


def _text(result: CallToolResult) -> str:
    parts = [
        item.text
        for item in result.content
        if isinstance(item, TextContent) and item.text.strip()
    ]
    return "\n".join(parts) if parts else "<no text>"


class _StatusAdapter:
    def __init__(self, agg: MCPAggregator) -> None:
        self._agg = agg
        self.config = SimpleNamespace(instruction="")

    async def get_server_status(self):
        return await self._agg.collect_server_status()


async def _status(agg: MCPAggregator, label: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    await render_mcp_status(_StatusAdapter(agg), indent="  ")


async def _call(agg: MCPAggregator, label: str, tool: str, args: dict) -> None:
    print(f"\n  ▶ {label}")
    try:
        result = await agg.call_tool(tool, args)
        for line in _text(result).splitlines():
            print(f"    {line}")
    except Exception as e:
        print(f"    ✗ Error: {e}")


async def _run_demo(
    title: str, name: str, script: str, steps: list[tuple[str, str, dict]]
) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")

    registry = ServerRegistry()
    registry.registry = {name: _settings(name, script)}
    context = Context(server_registry=registry)
    agg = MCPAggregator(
        server_names=[name],
        connection_persistence=True,
        context=context,
        name=f"demo-{name}",
    )

    async with agg:
        await _status(agg, f"{name}: after initialize")
        for label, tool, args in steps:
            await _call(agg, label, tool, args)
        await _status(agg, f"{name}: final")


# ---------------------------------------------------------------------------
# Each demo is a standalone async function run via subprocess
# ---------------------------------------------------------------------------

async def demo_session_required() -> None:
    await _run_demo(
        "DEMO 1: session-required (gatekeeper pattern)",
        "session-required",
        "session_required_server.py",
        [
            ("echo 'hello world'", "echo", {"text": "hello world"}),
            ("whoami", "whoami", {}),
            ("echo (session persists)", "echo", {"text": "session persists across calls!"}),
        ],
    )


async def demo_notebook() -> None:
    await _run_demo(
        "DEMO 2: notebook (per-session note storage)",
        "notebook",
        "notebook_server.py",
        [
            ("append: Buy milk", "notebook_append", {"text": "Buy milk"}),
            ("append: Write SEP draft", "notebook_append", {"text": "Write SEP draft"}),
            ("append: Review PR #2293", "notebook_append", {"text": "Review PR #2293"}),
            ("read all notes", "notebook_read", {}),
            ("notebook status", "notebook_status", {}),
            ("clear notebook", "notebook_clear", {}),
            ("read after clear", "notebook_read", {}),
        ],
    )


async def demo_hashcheck() -> None:
    await _run_demo(
        "DEMO 3: hashcheck (per-session hash KV store)",
        "hashcheck",
        "hashcheck_server.py",
        [
            ("store 'password'='secret123'", "hashcheck_store", {"key": "password", "text": "secret123"}),
            ("store 'api-key'='sk-abc'", "hashcheck_store", {"key": "api-key", "text": "sk-abc"}),
            ("list stored hashes", "hashcheck_list", {}),
            ("verify password (correct)", "hashcheck_verify", {"key": "password", "text": "secret123"}),
            ("verify password (WRONG)", "hashcheck_verify", {"key": "password", "text": "wrong-password"}),
            ("verify missing key", "hashcheck_verify", {"key": "nonexistent", "text": "anything"}),
            ("delete api-key", "hashcheck_delete", {"key": "api-key"}),
            ("list after delete", "hashcheck_list", {}),
        ],
    )


async def demo_selective() -> None:
    await _run_demo(
        "DEMO 4: selective-session (public + session-only tools)",
        "selective-session",
        "selective_session_server.py",
        [
            ("public echo (no session required)", "public_echo", {"text": "hello from public"}),
            ("reset current session", "session_reset", {}),
            (
                "session counter get (expected error: session missing)",
                "session_counter_get",
                {},
            ),
            ("start session explicitly", "session_start", {"label": "selective-demo"}),
            ("session counter increment", "session_counter_inc", {}),
            ("session counter get", "session_counter_get", {}),
        ],
    )


DEMOS = {
    "1": demo_session_required,
    "2": demo_notebook,
    "3": demo_hashcheck,
    "4": demo_selective,
    "session-required": demo_session_required,
    "notebook": demo_notebook,
    "hashcheck": demo_hashcheck,
    "selective": demo_selective,
}


async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Experimental MCP Sessions demo")
    parser.add_argument(
        "demo",
        nargs="?",
        default="all",
        choices=[
            "all",
            "1",
            "2",
            "3",
            "4",
            "session-required",
            "notebook",
            "hashcheck",
            "selective",
        ],
        help="Which demo to run (default: all — runs sequentially via subprocesses)",
    )
    args = parser.parse_args()

    if args.demo == "all":
        # Run each demo as a separate subprocess to avoid uvloop child-watcher issue
        for num in ["1", "2", "3", "4"]:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(Path(__file__)), num,
                stdout=None, stderr=None,  # inherit parent stdio
            )
            await proc.wait()
        print(f"\n{'═'*60}")
        print("  ALL DEMOS COMPLETE")
        print(f"{'═'*60}\n")
        return

    demo_fn = DEMOS[args.demo]
    try:
        await demo_fn()
    finally:
        await asyncio.sleep(0.05)
        await LoggingConfig.shutdown()
        await AsyncEventBus.get().stop()
        await asyncio.sleep(0.05)
        AsyncEventBus.reset()
        await asyncio.sleep(0.05)


if __name__ == "__main__":
    asyncio.run(main())
