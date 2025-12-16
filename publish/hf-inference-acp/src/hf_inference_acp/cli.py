"""Console entrypoint for hf-inference-acp.

This mirrors the `fast-agent-acp` option surface so tooling can invoke either
command interchangeably.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import sys
from pathlib import Path  # noqa: TC003 - typer needs runtime access

import typer

from fast_agent import FastAgent
from fast_agent.cli.commands import serve
from fast_agent.cli.commands.go import collect_stdio_commands, resolve_instruction_option
from fast_agent.cli.commands.server_helpers import add_servers_to_config, generate_server_name
from fast_agent.cli.commands.url_parser import generate_server_configs, parse_server_urls
from fast_agent.llm.model_factory import ModelFactory
from fast_agent.llm.provider_key_manager import ProviderKeyManager
from fast_agent.llm.provider_types import Provider
from fast_agent.mcp.hf_auth import add_hf_auth_header
from hf_inference_acp.agents import HuggingFaceAgent, SetupAgent
from hf_inference_acp.hf_config import (
    CONFIG_FILE,
    discover_hf_token,
    ensure_config_exists,
    get_default_model,
    has_hf_token,
    load_system_prompt,
)
from hf_inference_acp.wizard import WizardSetupLLM

# Register wizard-setup model locally
ModelFactory.MODEL_SPECIFIC_CLASSES["wizard-setup"] = WizardSetupLLM
ModelFactory.DEFAULT_PROVIDERS["wizard-setup"] = Provider.FAST_AGENT

app = typer.Typer(
    help="Run the Hugging Face Inference ACP agent over stdio.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)

ROOT_SUBCOMMANDS = {
    "go",
    "serve",
    "setup",
    "check",
    "auth",
    "bootstrap",
    "quickstart",
}


def get_setup_instruction() -> str:
    """Generate the instruction for the Setup agent."""
    token_status = "set" if has_hf_token() else "NOT SET"
    default_model = get_default_model()

    return f"""Hugging Face Inference Setup assistant.

# Available Commands

Use these slash commands to configure the agent:

- `/set-model <model>` - Set the default model for inference
- `/login` - Get instructions for logging in to HuggingFace
- `/check` - Verify huggingface_hub installation and configuration

# Current Status

- **Config file**: `{CONFIG_FILE}`
- **HF_TOKEN**: {token_status}
- **Default model**: `{default_model}`

To start using the AI assistant, ensure HF_TOKEN is set and switch to the "Hugging Face" mode."""


def get_hf_instruction() -> str:
    """Generate the instruction for the HuggingFace agent.

    Loads from ~/.config/hf-inference/hf.system_prompt.md so users can customize it.
    The file is created from the template on first run.
    """
    return load_system_prompt()


def _ensure_hf_token_from_provider_config(fast: FastAgent) -> None:
    """Set HF_TOKEN from the fast-agent provider config if it's not already set."""
    try:
        cfg = getattr(fast.app.context, "config", None)
        if cfg is None or os.environ.get("HF_TOKEN"):
            return
        # Prefer explicit provider config (fast-agent's Settings.hf.api_key)
        provider_token = ProviderKeyManager.get_config_file_key("hf", cfg)
        if provider_token:
            os.environ["HF_TOKEN"] = provider_token
    except Exception:
        # Best-effort; fall back to other discovery mechanisms
        return


def _ensure_hf_mcp_auth_header(fast: FastAgent) -> None:
    """Attach HF auth header to the huggingface MCP server if not already provided."""
    try:
        registry = getattr(fast.app.context, "server_registry", None)
        if registry is None:
            return

        server_config = registry.get_server_config("huggingface")
        if not server_config or not getattr(server_config, "url", None):
            return

        existing_headers = dict(server_config.headers or {})
        existing_keys = {k.lower() for k in existing_headers}
        if {"authorization", "x-hf-authorization"} & existing_keys:
            return

        updated_headers = add_hf_auth_header(server_config.url, existing_headers)
        if updated_headers is None:
            return

        server_config.headers = updated_headers
        registry.registry["huggingface"] = server_config
    except Exception:
        # Avoid breaking startup on header injection failure
        return


def _parse_stdio_servers(
    stdio_commands: list[str],
    server_list: list[str] | None,
) -> tuple[dict[str, dict[str, object]] | None, list[str] | None]:
    if not stdio_commands:
        return None, server_list

    stdio_servers: dict[str, dict[str, object]] = {}
    updated_server_list = list(server_list) if server_list else []

    for i, stdio_cmd in enumerate(stdio_commands):
        try:
            parsed_command = shlex.split(stdio_cmd)
            if not parsed_command:
                continue
            command = parsed_command[0]
            initial_args = parsed_command[1:] if len(parsed_command) > 1 else []

            if initial_args:
                for arg in initial_args:
                    if arg.endswith((".py", ".js", ".ts")):
                        base_name = generate_server_name(arg)
                        break
                else:
                    base_name = generate_server_name(command)
            else:
                base_name = generate_server_name(command)

            server_name = base_name
            if len(stdio_commands) > 1:
                server_name = f"{base_name}_{i + 1}"

            stdio_servers[server_name] = {
                "transport": "stdio",
                "command": command,
                "args": initial_args,
            }
            updated_server_list.append(server_name)
        except Exception:
            continue

    if not updated_server_list:
        updated_server_list = None

    return stdio_servers, updated_server_list


async def run_agents(
    *,
    name: str,
    config_path: str | None,
    server_list: list[str] | None,
    model: str | None,
    url_servers: dict[str, dict[str, str | dict[str, str]]] | None,
    stdio_servers: dict[str, dict[str, object]] | None,
    instruction_override: str | None,
    skills_directory: Path | None,
    shell_runtime: bool,
    host: str,
    port: int,
    tool_description: str | None,
    instance_scope: str,
    permissions_enabled: bool,
) -> None:
    """Main async function to set up and run the agents."""
    if config_path is None:
        config_path = str(ensure_config_exists())

    # Ensure HF_TOKEN is available for code paths that rely on it (e.g. MCP headers),
    # even when the user authenticated via `hf auth login`.
    if not os.environ.get("HF_TOKEN"):
        token, source = discover_hf_token(ignore_env=True)
        if token:
            os.environ["HF_TOKEN"] = token
            if source:
                os.environ["FAST_AGENT_HF_TOKEN_SOURCE"] = source

    default_model = get_default_model()
    effective_model = model or default_model

    # Create FastAgent instance
    fast_kwargs: dict[str, object] = {
        "name": name,
        "config_path": config_path,
        "parse_cli_args": False,
        "ignore_unknown_args": True,
        "quiet": True,
    }
    if skills_directory is not None:
        fast_kwargs["skills_directory"] = skills_directory

    fast = FastAgent(**fast_kwargs)

    if shell_runtime:
        await fast.app.initialize()
        setattr(fast.app.context, "shell_runtime", True)

    await add_servers_to_config(fast, url_servers or {})
    await add_servers_to_config(fast, stdio_servers or {})

    # Ensure HF_TOKEN is available from provider config for MCP auth
    _ensure_hf_token_from_provider_config(fast)
    hf_token_present = has_hf_token()

    # Register the Setup agent (wizard LLM for guided setup)
    # This is always available for configuration
    @fast.custom(
        SetupAgent,
        name="setup",
        instruction=get_setup_instruction(),
        model="wizard-setup",
        default=not hf_token_present,
    )
    async def setup_agent():
        pass

    instruction = instruction_override or get_hf_instruction()

    # Ensure huggingface is always in the servers list
    # (load_on_start in config controls whether it connects automatically)
    if server_list is None:
        server_list = []
    if "huggingface" not in server_list:
        server_list.append("huggingface")

    # Attach Authorization header to Hugging Face MCP server using HF_TOKEN if available
    if hf_token_present:
        _ensure_hf_mcp_auth_header(fast)

    # Register the HuggingFace agent (uses HF LLM)
    # Always register so the mode is visible; defaults to Setup mode when token is missing
    @fast.custom(
        HuggingFaceAgent,
        name="huggingface",
        instruction=instruction,
        model=effective_model if hf_token_present else "wizard-setup",
        servers=server_list,
        default=hf_token_present,
    )
    async def hf_agent():
        pass

    # Start the ACP server
    await fast.start_server(
        transport="acp",
        host=host,
        port=port,
        tool_description=tool_description,
        instance_scope=instance_scope,
        permissions_enabled=permissions_enabled,
    )


@app.callback(invoke_without_command=True, no_args_is_help=False)
def run_acp(
    ctx: typer.Context,
    name: str = typer.Option("hf-inference-acp", "--name", help="Name for the ACP server"),
    instruction: str | None = typer.Option(
        None, "--instruction", "-i", help="Path to file or URL containing instruction for the agent"
    ),
    config_path: str | None = typer.Option(None, "--config-path", "-c", help="Path to config file"),
    servers: str | None = typer.Option(
        None, "--servers", help="Comma-separated list of server names to enable from config"
    ),
    urls: str | None = typer.Option(
        None, "--url", help="Comma-separated list of HTTP/SSE URLs to connect to"
    ),
    auth: str | None = typer.Option(
        None, "--auth", help="Bearer token for authorization with URL-based servers"
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "--models",
        help="Override the default model (e.g., hf.moonshotai/Kimi-K2-Instruct-0905)",
    ),
    skills_dir: Path | None = typer.Option(
        None,
        "--skills-dir",
        "--skills",
        help="Override the default skills directory",
    ),
    npx: str | None = typer.Option(
        None, "--npx", help="NPX package and args to run as MCP server (quoted)"
    ),
    uvx: str | None = typer.Option(
        None, "--uvx", help="UVX package and args to run as MCP server (quoted)"
    ),
    stdio: str | None = typer.Option(
        None, "--stdio", help="Command to run as STDIO MCP server (quoted)"
    ),
    description: str | None = typer.Option(
        None,
        "--description",
        "-d",
        help="Description used for the exposed send tool (use {agent} to reference the agent name)",
    ),
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="Host address to bind when using HTTP or SSE transport",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        help="Port to use when running as a server with HTTP or SSE transport",
    ),
    shell: bool = typer.Option(
        False,
        "--shell",
        "-x",
        help="Enable a local shell runtime and expose the execute tool (bash or pwsh).",
    ),
    instance_scope: serve.InstanceScope = typer.Option(
        serve.InstanceScope.SHARED,
        "--instance-scope",
        help="Control how ACP clients receive isolated agent instances (shared, connection, request)",
    ),
    no_permissions: bool = typer.Option(
        False,
        "--no-permissions",
        help="Disable tool permission requests (allow all tool executions without asking)",
    ),
) -> None:
    stdio_commands = collect_stdio_commands(npx, uvx, stdio)
    shell_enabled = shell

    instruction_override = None
    if instruction:
        instruction_override, _ = resolve_instruction_option(instruction)

    server_list = servers.split(",") if servers else None

    url_servers = None
    if urls:
        try:
            parsed_urls = parse_server_urls(urls, auth)
            url_servers = generate_server_configs(parsed_urls)
            if url_servers and not server_list:
                server_list = list(url_servers.keys())
            elif url_servers and server_list:
                server_list.extend(list(url_servers.keys()))
        except ValueError as exc:
            print(f"Error parsing URLs: {exc}", file=sys.stderr)
            raise typer.Exit(1)

    stdio_servers, server_list = _parse_stdio_servers(stdio_commands, server_list)

    try:
        asyncio.run(
            run_agents(
                name=name,
                config_path=config_path,
                server_list=server_list,
                model=model,
                url_servers=url_servers,
                stdio_servers=stdio_servers,
                instruction_override=instruction_override,
                skills_directory=skills_dir,
                shell_runtime=shell_enabled,
                host=host,
                port=port,
                tool_description=description,
                instance_scope=instance_scope.value,
                permissions_enabled=not no_permissions,
            )
        )
    except KeyboardInterrupt:
        raise typer.Exit(0)


def main() -> None:
    """Console script entrypoint for `hf-inference-acp`."""
    # Match fast-agent-acp's behavior for consistent tooling integration.
    import click

    click.exceptions.UsageError.exit_code = 1

    args = sys.argv[1:]
    if args and args[0] in ROOT_SUBCOMMANDS:
        from fast_agent.cli.__main__ import main as root_cli_main

        root_cli_main()
        return
    app()


if __name__ == "__main__":
    main()
