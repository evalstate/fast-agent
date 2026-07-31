"""Request-building helpers for CLI runtime commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal
from urllib.parse import urlparse

import typer

from fast_agent.constants import DEFAULT_AGENT_INSTRUCTION
from fast_agent.core.agent_card_paths import is_agent_card_path
from fast_agent.mcp.connect_targets import (
    McpProtocolMode,
    NormalizedMcpTarget,
    build_server_config_from_target,
    infer_server_name,
    normalize_connect_target_text,
    redact_mcp_url,
    render_redacted_target,
    resolve_connect_auth_token,
)
from fast_agent.utils.collections import unique_preserve_order
from fast_agent.utils.commandline import split_commandline
from fast_agent.utils.text import strip_to_none

from .run_request import (
    AgentRunRequest,
    ExecutionMode,
    resolve_execution_mode,
)

if TYPE_CHECKING:
    from fast_agent.config import MCPServerSettings
    from fast_agent.llm.request_params import StructuredToolPolicy

DEFAULT_AGENT_CARDS_DIR: Final[Path] = Path(".fast-agent/agent-cards")
DEFAULT_TOOL_CARDS_DIR: Final[Path] = Path(".fast-agent/tool-cards")


@dataclass(frozen=True, slots=True)
class ResolvedInstructionOption:
    instruction: str
    agent_name: str


@dataclass(frozen=True, slots=True)
class StartupMcpMerge:
    servers: dict[str, MCPServerSettings] | None
    server_list: list[str] | None
    notices: tuple[str, ...]


def is_multi_model(model: str | None) -> bool:
    return bool(model and "," in model)


def resolve_default_instruction(
    model: str | None,
    mode: Literal["interactive", "serve"],
) -> str:
    del model, mode
    return DEFAULT_AGENT_INSTRUCTION


def merge_card_sources(
    sources: list[str] | None,
    default_dir: Path,
) -> list[str] | None:
    if sources:
        return unique_preserve_order(sources)

    merged: list[str] = []

    if default_dir.is_dir():
        has_cards = any(
            entry.is_file() and is_agent_card_path(entry) for entry in default_dir.iterdir()
        )
        if has_cards:
            merged.append(str(default_dir))

    return merged or None


def normalize_explicit_card_sources(sources: list[str] | None) -> list[str] | None:
    """Normalize explicit card sources without implicit directory scans."""
    if not sources:
        return None

    return unique_preserve_order(sources) or None


def _default_card_directories(home: Path | None) -> tuple[Path, Path]:
    from fast_agent.cli.command_support import get_settings_or_exit
    from fast_agent.paths import resolve_home_paths

    settings = get_settings_or_exit(home=home)
    home_paths = resolve_home_paths(settings=settings, override=home)
    return home_paths.agent_cards, home_paths.tool_cards


def validate_no_home_conflicts(
    *,
    no_home: bool,
    home: Path | None,
    resume: str | None,
) -> None:
    """Validate unsupported option combinations for --no-home mode."""
    if not no_home:
        return

    if home is not None:
        raise typer.BadParameter("Cannot combine --no-home with --home.")

    if resume is not None:
        raise typer.BadParameter("Cannot combine --no-home with --resume.")


def validate_shell_conflicts(*, shell_enabled: bool, no_shell: bool) -> None:
    """Validate unsupported shell option combinations."""
    if shell_enabled and no_shell:
        raise typer.BadParameter("Cannot combine --shell with --no-shell.")


def validate_subagent_conflicts(
    *,
    subagents: bool | None,
    subagent_model: str | None,
) -> None:
    """Reject an explicit disable paired with a fixed child model."""
    if subagents is False and subagent_model is not None:
        raise typer.BadParameter(
            "Cannot combine --subagent-model with --no-subagents.",
            param_hint="--subagent-model",
        )


def validate_execution_mode_inputs(
    *,
    message: str | None,
    prompt_file: str | None,
) -> ExecutionMode:
    try:
        return resolve_execution_mode(
            message=message,
            prompt_file=prompt_file,
        )
    except ValueError as exc:
        raise typer.BadParameter(
            str(exc),
            param_hint="--message/--prompt-file",
        ) from exc


def validate_attachment_inputs(
    *,
    attachments: list[str] | None,
    execution_mode: ExecutionMode,
) -> None:
    if attachments and execution_mode == "repl":
        raise typer.BadParameter(
            "--attach requires --message or --prompt-file",
            param_hint="--attach",
        )


def validate_json_schema_inputs(
    *,
    json_schema: str | None,
    schema_model: str | None = None,
    structured_tool_policy: str | None = None,
    execution_mode: ExecutionMode,
    model: str | None,
) -> StructuredToolPolicy | None:
    if json_schema is not None and schema_model is not None:
        raise typer.BadParameter(
            "Cannot combine --json-schema with --schema-model.",
            param_hint="--schema-model",
        )
    if structured_tool_policy is not None:
        from fast_agent.llm.request_params import is_structured_tool_policy

        if not is_structured_tool_policy(structured_tool_policy):
            raise typer.BadParameter(
                "structured tool policy must be 'auto', 'always', 'defer', or 'no_tools'",
                param_hint="--structured-tool-policy",
            )
        if schema_model is not None:
            raise typer.BadParameter(
                "--structured-tool-policy cannot be combined with --schema-model.",
                param_hint="--structured-tool-policy",
            )
        if json_schema is None:
            raise typer.BadParameter(
                "--structured-tool-policy requires --json-schema.",
                param_hint="--structured-tool-policy",
            )
        resolved_policy = structured_tool_policy
    else:
        resolved_policy = None
    if json_schema is None and schema_model is None:
        return resolved_policy
    if execution_mode == "repl":
        option = "--schema-model" if schema_model is not None else "--json-schema"
        raise typer.BadParameter(
            f"{option} requires --message or --prompt-file",
            param_hint=option,
        )
    if is_multi_model(model):
        option = "--schema-model" if schema_model is not None else "--json-schema"
        raise typer.BadParameter(
            f"Cannot combine {option} with multiple models.",
            param_hint=option,
        )
    return resolved_policy


def validate_multi_model_card_conflicts(
    *,
    model: str | None,
    merged_agent_cards: list[str] | None,
    merged_card_tools: list[str] | None,
    explicit_agent_cards: bool,
    explicit_card_tools: bool,
) -> None:
    """Reject unsupported combinations of multi-model mode and card loading."""
    if not is_multi_model(model):
        return

    if not merged_agent_cards and not merged_card_tools:
        return

    message = (
        "Cannot use multiple models with AgentCards or card tools. "
        "Multi-model mode (--model a,b) uses automatic parallel fan-out and requires no cards."
    )

    if explicit_agent_cards or explicit_card_tools:
        message += " Remove --agent-cards/--card-tool, or use a single --model value."
    else:
        message += (
            " Implicit cards were found in your environment; re-run with --no-home "
            "(or --home pointing to a directory without cards)."
        )

    raise typer.BadParameter(message, param_hint="--model")


def resolve_instruction_option(
    instruction: str | None,
    model: str | None,
    mode: Literal["interactive", "serve"],
) -> ResolvedInstructionOption:
    """Resolve the instruction option (file or URL) to text and inferred agent name."""
    resolved_instruction = resolve_default_instruction(model, mode)
    agent_name = "agent"

    if instruction:
        try:
            from fast_agent.core.instruction_source import resolve_instruction_source
            from fast_agent.io.source_resolver import REMOTE_TEXT_SCHEMES, materialize_text_source

            resolved_instruction = resolve_instruction_source(instruction)
            parsed_instruction = urlparse(str(instruction))
            if parsed_instruction.scheme not in REMOTE_TEXT_SCHEMES:
                instruction_path = materialize_text_source(instruction, label="instruction")
                if instruction_path.exists() and instruction_path.is_file():
                    agent_name = instruction_path.stem
        except Exception as exc:
            typer.echo(f"Error loading instruction from {instruction}: {exc}", err=True)
            raise typer.Exit(1) from exc

    return ResolvedInstructionOption(
        instruction=resolved_instruction,
        agent_name=agent_name,
    )


def collect_stdio_commands(npx: str | None, uvx: str | None, stdio: str | None) -> list[str]:
    """Collect STDIO command definitions from convenience options."""
    stdio_commands: list[str] = []

    if npx:
        stdio_commands.append(f"npx {npx}")
    if uvx:
        stdio_commands.append(f"uvx {uvx}")
    if stdio:
        stdio_commands.append(stdio)

    return stdio_commands


def resolve_instance_scope(
    *,
    transport: str,
    instance_scope: str | None,
) -> str:
    if transport == "acp":
        if instance_scope is None:
            return "connection"
        if instance_scope != "connection":
            raise ValueError(
                "ACP is always connection-scoped; --instance-scope must be omitted or set to connection."
            )
        return "connection"
    if instance_scope is None:
        return "shared"
    return instance_scope


def _expand_startup_urls(urls: list[str] | None) -> list[str]:
    expanded: list[str] = []
    for occurrence, value in enumerate(urls or (), start=1):
        for item in value.split(","):
            url = strip_to_none(item)
            if url is None:
                raise typer.BadParameter(
                    f"URL occurrence {occurrence} contains an empty value",
                    param_hint="--url",
                )
            expanded.append(url)
    return expanded


def _allocate_startup_server_name(base_name: str, selected: set[str]) -> str:
    if base_name not in selected:
        selected.add(base_name)
        return base_name
    suffix = 1
    while f"{base_name}_{suffix}" in selected:
        suffix += 1
    name = f"{base_name}_{suffix}"
    selected.add(name)
    return name


def _normalize_startup_stdio_target(command_text: str) -> NormalizedMcpTarget:
    try:
        tokens = split_commandline(command_text, syntax="posix")
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--npx/--uvx/--stdio") from exc
    if not tokens:
        raise typer.BadParameter(
            "STDIO command cannot be empty",
            param_hint="--npx/--uvx/--stdio",
        )
    if tokens[0] in {"npx", "uvx"}:
        return normalize_connect_target_text(command_text, syntax="posix")
    return NormalizedMcpTarget(
        mode="stdio",
        transport="stdio",
        url=None,
        command=tokens[0],
        args=tuple(tokens[1:]),
        server_name=None,
    )


def _materialize_startup_mcp_servers(
    *,
    server_list: list[str] | None,
    urls: list[str] | None,
    auth: str | None,
    client_metadata_url: str | None,
    stdio_commands: list[str] | None,
    protocol_mode: McpProtocolMode | None,
) -> StartupMcpMerge:
    url_values = _expand_startup_urls(urls)
    auth_token: str | None = None
    if auth is not None and url_values:
        try:
            auth_token = resolve_connect_auth_token(auth)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint="--auth") from exc

    targets: list[tuple[str, NormalizedMcpTarget, str | None, str | None]] = []
    for occurrence, url in enumerate(url_values, start=1):
        if not url.lower().startswith(("http://", "https://")):
            raise typer.BadParameter(
                f"URL occurrence {occurrence}: URL must have http or https scheme: "
                f"{redact_mcp_url(url)}",
                param_hint="--url",
            )
        try:
            target = normalize_connect_target_text(url, syntax="posix")
        except ValueError as exc:
            raise typer.BadParameter(
                f"URL occurrence {occurrence} is invalid: {redact_mcp_url(url)}",
                param_hint="--url",
            ) from exc
        if target.mode != "url":
            raise typer.BadParameter(
                f"URL occurrence {occurrence} must use http or https: {redact_mcp_url(url)}",
                param_hint="--url",
            )
        targets.append(("--url", target, auth_token, url))

    for command_text in stdio_commands or ():
        targets.append(
            (
                "--npx/--uvx/--stdio",
                _normalize_startup_stdio_target(command_text),
                None,
                None,
            )
        )

    if not targets:
        return StartupMcpMerge(servers=None, server_list=server_list, notices=())
    if auth_token is not None and client_metadata_url:
        raise typer.BadParameter(
            "Cannot combine --auth with --client-metadata-url for startup MCP URLs",
            param_hint="--auth",
        )

    selected_names = set(server_list or ())
    servers: dict[str, MCPServerSettings] = {}
    notices: list[str] = []
    common_overrides: dict[str, Any] = {}
    if protocol_mode is not None:
        common_overrides["protocol_mode"] = protocol_mode

    for source, target, target_auth, input_url in targets:
        server_name = _allocate_startup_server_name(infer_server_name(target), selected_names)
        overrides = dict(common_overrides)
        if target.mode == "url" and client_metadata_url:
            overrides["auth"] = {
                "oauth": True,
                "client_metadata_url": client_metadata_url,
            }
        try:
            built = build_server_config_from_target(
                target,
                server_name=server_name,
                auth_token=target_auth,
                overrides=overrides,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint=source) from exc
        servers[server_name] = built.settings
        resolved_target = render_redacted_target(target)
        if target.mode == "url" and input_url is not None and built.settings.url != input_url:
            resolved_target = redact_mcp_url(built.settings.url or target.url or input_url)
            notices.append(
                f"Resolved MCP URL '{redact_mcp_url(input_url)}' to '{resolved_target}'. "
                "Automatic '/mcp' suffixing is deprecated; pass the complete URL explicitly."
            )
        notices.append(f"Startup MCP server '{server_name}': {resolved_target}")

    merged_server_list = [*(server_list or ()), *servers]
    return StartupMcpMerge(
        servers=servers,
        server_list=merged_server_list,
        notices=tuple(notices),
    )


def build_agent_run_request(
    *,
    name: str,
    instruction: str,
    config_path: str | None,
    servers: str | None,
    urls: list[str] | None,
    auth: str | None,
    client_metadata_url: str | None,
    mcp_protocol: McpProtocolMode | None = None,
    agent_cards: list[str] | None,
    card_tools: list[str] | None,
    model: str | None,
    message: str | None,
    prompt_file: str | None,
    result_file: str | None,
    trajectory_output: Path | None = None,
    trajectory_format: Literal["atif"] = "atif",
    resume: str | None,
    stdio_commands: list[str] | None,
    agent_name: str | None,
    target_agent_name: str | None,
    skills_directory: Path | None,
    home: Path | None,
    shell_enabled: bool,
    mode: Literal["interactive", "serve"],
    transport: str,
    host: str,
    port: int,
    tool_description: str | None,
    tool_name_template: str | None,
    instance_scope: str | None,
    permissions_enabled: bool,
    reload: bool,
    watch: bool,
    quiet: bool = False,
    timeout_seconds: int | None = None,
    prefer_local_shell: bool = False,
    no_shell: bool = False,
    missing_shell_cwd_policy: Literal["ask", "create", "warn", "error"] | None = None,
    json_schema: str | None = None,
    schema_model: str | None = None,
    structured_tool_policy: str | None = None,
    no_home: bool = False,
    attachments: list[str] | None = None,
    environment: str | None = None,
    workspace: Path | None = None,
    subagents: bool | None = None,
    subagent_model: str | None = None,
) -> AgentRunRequest:
    """Build a normalized runtime request from legacy CLI kwargs."""
    validate_no_home_conflicts(
        no_home=no_home,
        home=home,
        resume=resume,
    )
    validate_shell_conflicts(shell_enabled=shell_enabled, no_shell=no_shell)
    validate_subagent_conflicts(subagents=subagents, subagent_model=subagent_model)
    execution_mode = validate_execution_mode_inputs(
        message=message,
        prompt_file=prompt_file,
    )
    validate_attachment_inputs(attachments=attachments, execution_mode=execution_mode)
    resolved_structured_tool_policy = validate_json_schema_inputs(
        json_schema=json_schema,
        schema_model=schema_model,
        structured_tool_policy=structured_tool_policy,
        execution_mode=execution_mode,
        model=model,
    )

    server_list = servers.split(",") if servers else None

    mcp_merge = _materialize_startup_mcp_servers(
        server_list=server_list,
        urls=urls,
        auth=auth,
        client_metadata_url=client_metadata_url,
        stdio_commands=stdio_commands,
        protocol_mode=mcp_protocol,
    )
    server_list = mcp_merge.server_list

    if no_home:
        merged_agent_cards = normalize_explicit_card_sources(agent_cards)
        merged_card_tools = normalize_explicit_card_sources(card_tools)
    else:
        default_agent_cards_dir, default_tool_cards_dir = _default_card_directories(home)
        merged_agent_cards = merge_card_sources(agent_cards, default_agent_cards_dir)
        merged_card_tools = merge_card_sources(card_tools, default_tool_cards_dir)

    validate_multi_model_card_conflicts(
        model=model,
        merged_agent_cards=merged_agent_cards,
        merged_card_tools=merged_card_tools,
        explicit_agent_cards=bool(agent_cards),
        explicit_card_tools=bool(card_tools),
    )

    effective_permissions_enabled = (
        permissions_enabled if not (no_home and mode == "serve") else False
    )
    effective_shell_enabled = shell_enabled or (environment is not None and not no_shell)

    return AgentRunRequest(
        name=name,
        instruction=instruction,
        config_path=config_path,
        server_list=server_list,
        agent_cards=merged_agent_cards,
        card_tools=merged_card_tools,
        model=model,
        message=message,
        prompt_file=prompt_file,
        attachments=attachments,
        json_schema=json_schema,
        schema_model=schema_model,
        structured_tool_policy=resolved_structured_tool_policy,
        result_file=result_file,
        trajectory_output=trajectory_output,
        trajectory_format=trajectory_format,
        resume=resume,
        startup_mcp_servers=mcp_merge.servers,
        mcp_startup_notices=mcp_merge.notices,
        agent_name=agent_name,
        target_agent_name=target_agent_name,
        environment=environment,
        skills_directory=skills_directory,
        home=None if no_home else home,
        workspace=workspace,
        no_home=no_home,
        shell_runtime=effective_shell_enabled,
        no_shell=no_shell,
        prefer_local_shell=prefer_local_shell,
        mode=mode,
        transport=transport,
        host=host,
        port=port,
        tool_description=tool_description,
        tool_name_template=tool_name_template,
        instance_scope=resolve_instance_scope(
            transport=transport,
            instance_scope=instance_scope,
        ),
        permissions_enabled=effective_permissions_enabled,
        reload=reload,
        watch=watch,
        execution_mode=execution_mode,
        quiet=quiet,
        timeout_seconds=timeout_seconds,
        missing_shell_cwd_policy=missing_shell_cwd_policy,
        subagents=True if subagent_model is not None else subagents,
        subagent_model=subagent_model,
    )


def build_run_agent_kwargs(
    request: AgentRunRequest | None = None,
    /,
    **request_kwargs: Any,
) -> dict[str, Any]:
    if request is None:
        request_kwargs.setdefault("attachments", None)
        request = build_agent_run_request(**request_kwargs)
    return request.to_agent_setup_kwargs()


def build_command_run_request(
    *,
    name: str,
    instruction_option: str | None,
    config_path: str | None,
    servers: str | None,
    urls: list[str] | None,
    auth: str | None,
    client_metadata_url: str | None,
    mcp_protocol: McpProtocolMode | None = None,
    agent_cards: list[str] | None,
    card_tools: list[str] | None,
    model: str | None,
    message: str | None,
    prompt_file: str | None,
    result_file: str | None,
    trajectory_output: Path | None = None,
    trajectory_format: Literal["atif"] = "atif",
    resume: str | None,
    npx: str | None,
    uvx: str | None,
    stdio: str | None,
    target_agent_name: str | None,
    skills_directory: Path | None,
    home: Path | None,
    shell_enabled: bool,
    mode: Literal["interactive", "serve"],
    transport: str = "http",
    host: str = "127.0.0.1",
    port: int = 8000,
    tool_description: str | None = None,
    tool_name_template: str | None = None,
    instance_scope: str | None = None,
    permissions_enabled: bool = True,
    reload: bool = False,
    watch: bool = False,
    quiet: bool = False,
    timeout_seconds: int | None = None,
    prefer_local_shell: bool = False,
    no_shell: bool = False,
    missing_shell_cwd_policy: Literal["ask", "create", "warn", "error"] | None = None,
    no_home: bool = False,
    json_schema: str | None = None,
    schema_model: str | None = None,
    structured_tool_policy: str | None = None,
    attachments: list[str] | None = None,
    environment: str | None = None,
    workspace: Path | None = None,
    subagents: bool | None = None,
    subagent_model: str | None = None,
) -> AgentRunRequest:
    """Build a normalized request directly from command option values."""
    validate_no_home_conflicts(
        no_home=no_home,
        home=home,
        resume=resume,
    )
    validate_shell_conflicts(shell_enabled=shell_enabled, no_shell=no_shell)
    validate_subagent_conflicts(subagents=subagents, subagent_model=subagent_model)

    stdio_commands = collect_stdio_commands(npx, uvx, stdio)
    resolved_instruction = resolve_instruction_option(
        instruction_option,
        model,
        mode,
    )

    return build_agent_run_request(
        name=name,
        instruction=resolved_instruction.instruction,
        config_path=config_path,
        servers=servers,
        urls=urls,
        auth=auth,
        client_metadata_url=client_metadata_url,
        mcp_protocol=mcp_protocol,
        agent_cards=agent_cards,
        card_tools=card_tools,
        model=model,
        message=message,
        prompt_file=prompt_file,
        attachments=attachments,
        json_schema=json_schema,
        schema_model=schema_model,
        structured_tool_policy=structured_tool_policy,
        result_file=result_file,
        trajectory_output=trajectory_output,
        trajectory_format=trajectory_format,
        resume=resume,
        stdio_commands=stdio_commands,
        agent_name=resolved_instruction.agent_name,
        target_agent_name=target_agent_name,
        environment=environment,
        skills_directory=skills_directory,
        home=home,
        workspace=workspace,
        no_home=no_home,
        shell_enabled=shell_enabled,
        prefer_local_shell=prefer_local_shell,
        no_shell=no_shell,
        mode=mode,
        transport=transport,
        host=host,
        port=port,
        tool_description=tool_description,
        tool_name_template=tool_name_template,
        instance_scope=instance_scope,
        permissions_enabled=permissions_enabled,
        reload=reload,
        watch=watch,
        quiet=quiet,
        timeout_seconds=timeout_seconds,
        missing_shell_cwd_policy=missing_shell_cwd_policy,
        subagents=subagents,
        subagent_model=subagent_model,
    )
