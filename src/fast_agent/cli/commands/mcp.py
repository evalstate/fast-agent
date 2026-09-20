"""Read-only MCP diagnostics using the production connection lifecycle."""

from __future__ import annotations

import asyncio
import json
import math
import os
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Literal

import typer

from fast_agent.config import Settings, get_settings
from fast_agent.context import Context
from fast_agent.core.exceptions import walk_exception_chain
from fast_agent.mcp.client_callback_runtime import MCPClientCallbackRuntime
from fast_agent.mcp.client_gateway import is_http_auth_challenge
from fast_agent.mcp.failures import safe_mcp_diagnostic_text, safe_mcp_exception_text
from fast_agent.mcp.mcp_connection_manager import MCPConnectionManager
from fast_agent.mcp_server_registry import ServerRegistry

if TYPE_CHECKING:
    from mcp_types import (
        ListPromptsResult,
        ListResourcesResult,
        ListResourceTemplatesResult,
        ListToolsResult,
    )

    from fast_agent.mcp.client_connection import MCPClientConnection
    from fast_agent.mcp.oauth_client import OAuthEvent

app = typer.Typer(help="Diagnose configured MCP servers without an LLM.", no_args_is_help=True)


type DiscoveryKind = Literal["tools", "prompts", "resources", "resource_templates"]


class DiagnosticCallbacks(MCPClientCallbackRuntime):
    def __post_init__(self) -> None:
        super().__post_init__()
        # Manager clones callback runtimes; enforce this on every clone as well.
        self.sampling_callback = None
        self.sampling_capabilities = None
        self.elicitation_callback = None


@dataclass
class Phase:
    state: str = "pending"
    seconds: float = 0.0
    count: int = 0
    pages: int = 0
    cache: str = "not_accessed"
    error: str | None = None


@dataclass
class Report:
    server: str
    ok: bool = False
    protocol: str | None = None
    negotiation: str | None = None
    capabilities: dict[str, bool] = field(default_factory=dict)
    oauth_events: list[str] = field(default_factory=list)
    capability_cache: str = "empty"
    phases: dict[str, Phase] = field(default_factory=dict)


def _safe_error(exc: Exception) -> str:
    detail = safe_mcp_exception_text(exc)
    chain = list(walk_exception_chain(exc))
    if any(isinstance(cause, TimeoutError) for cause in chain):
        return f"{detail}\nTimed out. Check server reachability or increase --timeout."
    if is_http_auth_challenge(exc):
        return f"{detail}\nAuthentication required. Run fast-agent auth mcp login SERVER."
    if any(isinstance(cause, FileNotFoundError) for cause in chain):
        return f"{detail}\nExecutable not found. Check the configured command and PATH."
    return (
        f"{detail}\nRequest failed. Check fast-agent check and fast-agent auth mcp; "
        "verify the server command, endpoint, credentials and protocol_mode."
    )


async def _discover(client: MCPClientConnection, kind: DiscoveryKind, phase: Phase) -> None:
    cursor: str | None = None
    seen: set[str] = set()
    phase.cache = "refresh"
    while True:
        result: (
            ListToolsResult | ListPromptsResult | ListResourcesResult | ListResourceTemplatesResult
        )
        if kind == "tools":
            result = await client.list_tools(cursor=cursor, cache_mode="refresh")
            count = len(result.tools)
        elif kind == "prompts":
            result = await client.list_prompts(cursor=cursor, cache_mode="refresh")
            count = len(result.prompts)
        elif kind == "resources":
            result = await client.list_resources(cursor=cursor, cache_mode="refresh")
            count = len(result.resources)
        else:
            result = await client.list_resource_templates(cursor=cursor, cache_mode="refresh")
            count = len(result.resource_templates)
        phase.pages += 1
        phase.count += count
        cursor = result.next_cursor
        if cursor is None:
            return
        if cursor in seen:
            raise ValueError("Repeated discovery cursor")
        seen.add(cursor)


async def diagnose_server(settings: Settings, server: str, timeout: float) -> Report:
    report = Report(server=safe_mcp_diagnostic_text(server))
    registry = ServerRegistry(config=settings)
    config = registry.registry.get(server)
    if config is None:
        report.phases["configuration"] = Phase(
            state="error", error="Unknown server. Check mcp.servers in fast-agent configuration."
        )
        return report
    context = Context(config=settings, server_registry=registry)
    callbacks = DiagnosticCallbacks(server_name=server, server_config=config, context=context)
    manager = MCPConnectionManager(registry, context=context)
    active = Phase(state="running")
    report.phases["connection"] = active
    started = perf_counter()

    async def on_oauth_event(event: OAuthEvent) -> None:
        # Suppress OAuth console output; retain only fixed event kinds, never URLs/messages.
        report.oauth_events.append(event.event_type)

    await manager.__aenter__()
    try:
        async with asyncio.timeout(timeout):
            conn = await manager.get_server(
                server,
                callback_runtime=callbacks,
                startup_timeout_seconds=timeout,
                allow_oauth_paste_fallback=False,
                oauth_event_handler=on_oauth_event,
            )
            active.state = "ok"
            active.seconds = perf_counter() - started
            report.protocol = safe_mcp_diagnostic_text(conn.protocol_version or "")
            report.negotiation = safe_mcp_diagnostic_text(conn.negotiation or "")
            caps = conn.server_capabilities
            report.capability_cache = (
                "populated" if registry.get_server_capabilities(server) is not None else "empty"
            )
            kinds: dict[DiscoveryKind, bool] = {
                "tools": caps is not None and caps.tools is not None,
                "prompts": caps is not None and caps.prompts is not None,
                "resources": caps is not None and caps.resources is not None,
                "resource_templates": caps is not None and caps.resources is not None,
            }
            report.capabilities = dict(kinds)
            if caps is not None:
                report.capabilities.update(
                    {
                        "logging": caps.logging is not None,
                        "completions": caps.completions is not None,
                        "tasks": caps.tasks is not None,
                        "experimental": bool(caps.experimental),
                        "extensions": bool(caps.extensions),
                    }
                )
            assert conn.client is not None
            for kind, supported in kinds.items():
                active = Phase(state="running" if supported else "unsupported")
                report.phases[kind] = active
                if not supported:
                    continue
                started = perf_counter()
                try:
                    await _discover(conn.client, kind, active)
                    active.state = "ok"
                except Exception as exc:
                    active.state = "error"
                    active.error = _safe_error(exc)
                finally:
                    active.seconds = perf_counter() - started
    except Exception as exc:
        active.state = "timeout" if isinstance(exc, TimeoutError) else "error"
        active.seconds = perf_counter() - started
        active.error = _safe_error(exc)
    finally:
        cleanup = Phase(state="running")
        report.phases["cleanup"] = cleanup
        started = perf_counter()
        # Cancel blocked lifecycle tasks before awaiting the manager's task group.
        for connection in manager.running_servers.values():
            connection.cancel_lifecycle()
        try:
            async with asyncio.timeout(5):
                await manager.__aexit__(None, None, None)
            cleanup.state = "ok"
        except Exception as exc:
            cleanup.state = "error"
            cleanup.error = _safe_error(exc)
        cleanup.seconds = perf_counter() - started
    report.ok = all(p.state in {"ok", "unsupported"} for p in report.phases.values())
    return report


@app.callback()
def main() -> None:
    """MCP inspection commands."""


@app.command("diagnose")
def diagnose(
    server: str = typer.Argument(..., help="Configured mcp.servers name (not a URL)."),
    json_output: bool = typer.Option(False, "--json", help="Emit one JSON report."),
    timeout: float = typer.Option(
        30.0, "--timeout", help="Total connection/discovery budget in seconds."
    ),
    config_path: str | None = typer.Option(None, "--config", "-c", help="Configuration file."),
) -> None:
    """Connect and list tools, prompts, resources and templates; never invoke tools."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise typer.BadParameter("must be finite and greater than zero", param_hint="--timeout")
    # Keep SDK/auth/server output (which may contain credentials) out of both report streams.
    with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
        try:
            settings = get_settings(config_path)
            report = asyncio.run(diagnose_server(settings, server, timeout))
        except Exception as exc:
            report = Report(
                server=safe_mcp_diagnostic_text(server),
                phases={"configuration": Phase(state="error", error=_safe_error(exc))},
            )
    if json_output:
        typer.echo(json.dumps(asdict(report), indent=2))
    else:
        typer.echo(f"MCP {report.server}: {'OK' if report.ok else 'FAILED'}")
        typer.echo(
            f"Protocol: {report.protocol or 'unavailable'} ({report.negotiation or 'unknown'})"
        )
        typer.echo(
            f"Capabilities: {json.dumps(report.capabilities)}; cache: {report.capability_cache}"
        )
        for name, phase in report.phases.items():
            typer.echo(
                f"{name}: {phase.state} ({phase.seconds:.3f}s), "
                f"{phase.count} items / {phase.pages} pages, cache={phase.cache}"
            )
            if phase.error:
                typer.echo(f"  {phase.error}")
    raise typer.Exit(0 if report.ok else 1)
