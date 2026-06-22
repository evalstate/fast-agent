"""Server-mode runtime dispatch for FastAgent."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from fast_agent.ui.console import configure_console_stream
from fast_agent.utils.transports import uses_protocol_stdio

if TYPE_CHECKING:
    from fast_agent.config import Settings
    from fast_agent.core.fastagent import ManagedRunState, RunSettings, RuntimeCallbacks


class ACPServerFactory(Protocol):
    def __call__(self) -> type[Any]: ...


@dataclass(frozen=True, slots=True)
class ServerRuntimeContext:
    app_name: str
    args: Any
    config: "Settings | None"
    skills_directory_override: Any
    state: "ManagedRunState"
    callbacks: "RuntimeCallbacks"
    settings: "RunSettings"
    acp_server_factory: ACPServerFactory


def print_server_startup(
    *,
    app_name: str,
    args: Any,
    output_stream: Any,
) -> None:
    print(f"Starting fast-agent  '{app_name}' in server mode", file=output_stream)
    print(f"Transport: {args.transport}", file=output_stream)
    if args.transport in {"http", "a2a"}:
        print(f"Listening on {args.host}:{args.port}", file=output_stream)
    print("Press Ctrl+C to stop", file=output_stream)


def resolve_server_instance_scope(
    *,
    transport: str,
    instance_scope: str | None,
) -> str:
    if transport == "acp":
        if instance_scope is None:
            return "connection"
        if instance_scope != "connection":
            raise ValueError(
                "ACP is always connection-scoped; instance_scope must be omitted or set to 'connection'."
            )
        return "connection"
    if instance_scope is None:
        return "shared"
    return instance_scope


async def run_server_mode(context: ServerRuntimeContext) -> None:
    settings = context.settings
    if not settings.server_mode:
        return

    is_stdio_transport = uses_protocol_stdio(settings.transport)
    configure_console_stream("stderr" if is_stdio_transport else "stdout")
    output_stream = sys.stderr if is_stdio_transport else sys.stdout

    try:
        if not settings.quiet_mode:
            print_server_startup(
                app_name=context.app_name,
                args=context.args,
                output_stream=output_stream,
            )

        if settings.transport == "acp":
            await run_acp_server(context)
        elif settings.transport == "a2a":
            await run_a2a_server(context)
        else:
            await run_mcp_server(context)
    except KeyboardInterrupt:
        if not settings.quiet_mode:
            print("\nServer stopped by user (Ctrl+C)", file=output_stream)
        raise SystemExit(0) from None
    except Exception as exc:
        if not settings.quiet_mode:
            import traceback

            traceback.print_exc()
        print(f"\nServer stopped with error: {exc}", file=output_stream)
        raise SystemExit(1) from exc

    raise SystemExit(0)


async def run_acp_server(context: ServerRuntimeContext) -> None:
    AgentACPServer = context.acp_server_factory()

    server_name = getattr(context.args, "server_name", None)
    resolve_server_instance_scope(
        transport="acp",
        instance_scope=getattr(context.args, "instance_scope", None),
    )
    permissions_enabled = getattr(context.args, "permissions_enabled", True)

    acp_server = AgentACPServer(
        bootstrap_instance=context.state.primary_instance,
        create_instance=context.callbacks.create_instance,
        dispose_instance=context.callbacks.dispose_instance,
        server_name=server_name or f"{context.app_name}",
        skills_directory_override=context.skills_directory_override,
        permissions_enabled=permissions_enabled,
        load_card_callback=context.callbacks.load_card_source,
        attach_agent_tools_callback=context.callbacks.attach_agent_tools_source,
        detach_agent_tools_callback=context.callbacks.detach_agent_tools_source,
        dump_agent_card_callback=context.callbacks.dump_agent_card,
        reload_callback=context.callbacks.reload_source,
    )
    await acp_server.run_async()


async def run_mcp_server(context: ServerRuntimeContext) -> None:
    from fast_agent.mcp.server.harness_app_server import (
        HarnessMCPAppRuntimeOptions,
        run_harness_mcp_app_server,
    )

    server_name = getattr(context.args, "server_name", None)
    await run_harness_mcp_app_server(
        instance_factory=context.callbacks.instance_factory(),
        shell_executor=context.state.runtime.shell_executor,
        settings=context.config,
        options=HarnessMCPAppRuntimeOptions(
            server_name=server_name or f"{context.app_name}-MCP-Server",
            server_description=getattr(context.args, "server_description", None),
            tool_description=getattr(context.args, "tool_description", None),
            default_agent=getattr(context.args, "agent", None),
            transport=context.args.transport,
            host=context.args.host,
            port=context.args.port,
            instance_scope=getattr(context.args, "instance_scope", "shared"),
        ),
    )


async def run_a2a_server(context: ServerRuntimeContext) -> None:
    from fast_agent.a2a import AgentA2AServer

    server_description = getattr(context.args, "server_description", None)
    server_name = getattr(context.args, "server_name", None)
    instance_scope = getattr(context.args, "instance_scope", "shared")
    a2a_server = AgentA2AServer(
        primary_instance=context.state.primary_instance,
        create_instance=context.callbacks.create_instance,
        dispose_instance=context.callbacks.dispose_instance,
        server_name=server_name or f"{context.app_name}",
        server_description=server_description,
        host=context.args.host,
        port=context.args.port,
        instance_scope=instance_scope,
    )
    await a2a_server.run_async(host=context.args.host, port=context.args.port)


__all__ = [
    "ServerRuntimeContext",
    "print_server_startup",
    "resolve_server_instance_scope",
    "run_a2a_server",
    "run_acp_server",
    "run_mcp_server",
    "run_server_mode",
]
