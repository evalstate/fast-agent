"""Interactive CLI helpers for model alias setup."""

from __future__ import annotations

import asyncio
import shlex
import sys
from pathlib import Path
from typing import Literal

import typer
from pydantic import ValidationError
from rich.text import Text

from fast_agent.cli.env_helpers import resolve_environment_dir_option
from fast_agent.cli.shared_options import CommonAgentOptions
from fast_agent.commands.context import AgentProvider, CommandContext, CommandIO
from fast_agent.commands.handlers import models_manager
from fast_agent.commands.results import CommandMessage, CommandOutcome
from fast_agent.config import (
    Settings,
    deep_merge,
    find_fastagent_config_files,
    load_layered_settings,
    load_yaml_mapping,
    resolve_config_search_root,
)
from fast_agent.llm.model_alias_diagnostics import (
    ModelAliasSetupDiagnostics,
    ModelAliasSetupItem,
    collect_model_alias_setup_diagnostics,
)
from fast_agent.ui.adapters.tui_io import TuiCommandIO
from fast_agent.ui.model_alias_picker import (
    CUSTOM_ALIAS_SENTINEL,
    run_model_alias_picker_async,
)

type WriteTarget = Literal["env", "project"]

app = typer.Typer(help="Interactive model alias setup.")


class _CliModelAgentProvider(AgentProvider):
    """Minimal provider used for CLI-only command contexts."""

    def _agent(self, name: str) -> object:
        raise KeyError(name)

    def agent_names(self) -> list[str]:
        return []

    async def list_prompts(
        self,
        namespace: str | None,
        agent_name: str | None = None,
    ) -> object:
        del namespace, agent_name
        return {}


def _build_alias_setup_argument(
    *,
    token: str | None,
    target: WriteTarget,
    dry_run: bool,
) -> str:
    parts = ["set"]
    if token is not None and token.strip():
        parts.append(shlex.quote(token.strip()))
    parts.extend(["--target", target])
    if dry_run:
        parts.append("--dry-run")
    return " ".join(parts)


def _normalize_write_target(value: str) -> WriteTarget:
    normalized = value.strip().lower()
    if normalized == "env":
        return "env"
    if normalized == "project":
        return "project"
    raise typer.BadParameter("--target must be either 'env' or 'project'.")


async def run_model_setup(
    *,
    io: CommandIO,
    settings: Settings,
    token: str | None,
    target: WriteTarget = "env",
    dry_run: bool = False,
) -> CommandOutcome:
    """Execute the shared interactive alias-setup flow."""
    resolved_token = token
    if resolved_token is None:
        diagnostics = collect_model_alias_setup_diagnostics(
            cwd=Path.cwd(),
            env_dir=getattr(settings, "environment_dir", None),
        )
        resolved_token = await _select_model_setup_token(
            io,
            diagnostics=diagnostics,
        )

    provider = _CliModelAgentProvider()
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="cli",
        io=io,
        settings=settings,
    )
    argument = _build_alias_setup_argument(
        token=resolved_token,
        target=target,
        dry_run=dry_run,
    )
    return await models_manager.handle_models_command(
        ctx,
        agent_name="cli",
        action="aliases",
        argument=argument,
    )


async def _select_model_setup_token(
    io: CommandIO,
    *,
    diagnostics: ModelAliasSetupDiagnostics,
) -> str | None:
    items = diagnostics.items
    common_items = _build_common_setup_items(diagnostics.valid_aliases)
    if not items:
        if isinstance(io, TuiCommandIO) and common_items:
            return await run_model_alias_picker_async(common_items)
        return None

    if isinstance(io, TuiCommandIO):
        selected_token = await run_model_alias_picker_async(
            _merge_setup_items(items, common_items)
        )
        if selected_token == CUSTOM_ALIAS_SENTINEL:
            return await io.prompt_text(
                "Alias token ($namespace.key):",
                allow_empty=False,
            )
        if selected_token is not None:
            return selected_token
        return None

    if len(items) == 1:
        item = items[0]
        await io.emit(
            CommandMessage(
                text=_render_setup_item_summary(
                    item,
                    title="Detected one alias that needs setup",
                ),
                right_info="model",
            )
        )
        return item.token

    await io.emit(
        CommandMessage(
            text=_render_setup_item_list(items),
            right_info="model",
        )
    )
    option_labels = {
        str(index): item.token
        for index, item in enumerate(items, start=1)
    }
    selection = await io.prompt_selection(
        "Alias to configure (number or 'custom'):",
        options=[*option_labels.keys(), "custom"],
        allow_cancel=True,
    )
    if selection is None:
        return None

    normalized_selection = selection.strip().lower()
    if normalized_selection == "custom":
        return await io.prompt_text(
            "Alias token ($namespace.key):",
            allow_empty=False,
        )
    return option_labels.get(normalized_selection)


def _render_setup_item_summary(item: ModelAliasSetupItem, *, title: str) -> Text:
    content = Text()
    content.append(f"{title}\n", style="bold")
    content.append(f"• {item.token}\n", style="cyan")
    content.append(f"  {item.priority}/{item.status}: {item.summary}\n", style="yellow")
    if item.references:
        content.append(
            f"  used by: {', '.join(item.references)}",
            style="dim",
        )
    return content


def _render_setup_item_list(items: tuple[ModelAliasSetupItem, ...]) -> Text:
    content = Text()
    content.append("Aliases that need setup\n", style="bold")
    for index, item in enumerate(items, start=1):
        content.append(
            f"{index}. {item.token}  [{item.priority}/{item.status}]\n",
            style="cyan" if item.priority == "recommended" else "yellow",
        )
        content.append(f"   {item.summary}\n", style="white")
        if item.references:
            content.append(
                f"   used by: {', '.join(item.references)}\n",
                style="dim",
            )
        if item.current_value is not None:
            current_value = item.current_value if item.current_value else "<empty>"
            content.append(f"   current: {current_value}\n", style="dim")
    content.append("\nType 'custom' to enter a different alias token.", style="dim")
    return content


def _build_common_setup_items(
    valid_aliases: dict[str, dict[str, str]],
) -> tuple[ModelAliasSetupItem, ...]:
    items: list[ModelAliasSetupItem] = []
    system_aliases = valid_aliases.get("system", {})
    if "default" not in system_aliases:
        items.append(
            ModelAliasSetupItem(
                token="$system.default",
                priority="required",
                status="missing",
                current_value=None,
                summary="Recommended starter alias for your main default model.",
                references=("starter setup",),
            )
        )
    if "fast" not in system_aliases:
        items.append(
            ModelAliasSetupItem(
                token="$system.fast",
                priority="recommended",
                status="missing",
                current_value=None,
                summary="Optional starter alias for a faster or cheaper model.",
                references=("starter setup",),
            )
        )
    return tuple(items)


def _merge_setup_items(
    primary_items: tuple[ModelAliasSetupItem, ...],
    extra_items: tuple[ModelAliasSetupItem, ...],
) -> tuple[ModelAliasSetupItem, ...]:
    merged: list[ModelAliasSetupItem] = list(primary_items)
    seen_tokens = {item.token for item in primary_items}
    for item in extra_items:
        if item.token in seen_tokens:
            continue
        merged.append(item)
    return tuple(merged)


async def _run_model_setup_command(
    *,
    settings: Settings,
    token: str | None,
    target: WriteTarget,
    dry_run: bool,
) -> None:
    config_payload = _load_tolerant_config_payload(
        cwd=Path.cwd(),
        env_dir=getattr(settings, "environment_dir", None),
    )
    provider = _CliModelAgentProvider()
    io = TuiCommandIO(
        prompt_provider=provider,
        agent_name="cli",
        settings=settings,
        config_payload=config_payload,
    )
    outcome = await run_model_setup(
        io=io,
        settings=settings,
        token=token,
        target=target,
        dry_run=dry_run,
    )
    for message in outcome.messages:
        await io.emit(message)


def _load_tolerant_config_payload(
    *,
    cwd: Path,
    env_dir: str | Path | None,
) -> dict[str, object] | None:
    try:
        merged_settings, _ = load_layered_settings(start_path=cwd, env_dir=env_dir)
        search_root = resolve_config_search_root(cwd, env_dir=env_dir)
        _, secrets_path = find_fastagent_config_files(search_root)
        if secrets_path and secrets_path.exists():
            merged_settings = deep_merge(merged_settings, load_yaml_mapping(secrets_path))
    except Exception:
        return None
    return merged_settings or None


def _print_validation_error(exc: ValidationError) -> None:
    typer.echo("fast-agent model setup could not load the current configuration.", err=True)
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        message = error.get("msg", "invalid value")
        if location:
            typer.echo(f"  - {location}: {message}", err=True)
        else:
            typer.echo(f"  - {message}", err=True)
    typer.echo("Hint: run `fast-agent check` for a broader config report.", err=True)


@app.callback(invoke_without_command=True)
def model_main(ctx: typer.Context) -> None:
    """Manage interactive model setup flows."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(0)


@app.command("setup")
def model_setup(
    ctx: typer.Context,
    token: str | None = typer.Argument(
        None,
        help="Alias token to update, such as $system.fast. Omit to choose or create one interactively.",
    ),
    env: str | None = CommonAgentOptions.env_dir(),
    target: str = typer.Option(
        "env",
        "--target",
        help="Where to save alias changes.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview changes without writing files.",
    ),
) -> None:
    """Interactively create or update a model alias using the model selector."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        typer.echo("fast-agent model setup requires an interactive terminal.", err=True)
        raise typer.Exit(1)

    resolved_env_dir = resolve_environment_dir_option(
        ctx,
        Path(env) if env is not None else None,
    )
    resolved_target = _normalize_write_target(target)
    settings = (
        Settings(environment_dir=str(resolved_env_dir))
        if resolved_env_dir is not None
        else Settings()
    )

    try:
        asyncio.run(
            _run_model_setup_command(
                settings=settings,
                token=token,
                target=resolved_target,
                dry_run=dry_run,
            )
        )
    except ValidationError as exc:
        _print_validation_error(exc)
        raise typer.Exit(1) from exc
