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
    ModelAliasPickerItem,
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


def _normalize_interactive_alias_token(token: str | None) -> str | None:
    if token is None:
        return None
    stripped = token.strip()
    if not stripped:
        return stripped
    if stripped.startswith("$"):
        return stripped
    return f"${stripped}"


async def _prompt_manual_alias_token(io: CommandIO) -> str | None:
    return _normalize_interactive_alias_token(
        await io.prompt_text(
            "Alias token ($namespace.key):",
            allow_empty=False,
        )
    )


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
        has_guided_choices = bool(diagnostics.items) or (
            isinstance(io, TuiCommandIO)
            and bool(_build_common_setup_items(diagnostics.valid_aliases))
        )
        resolved_token = await _select_model_setup_token(
            io,
            diagnostics=diagnostics,
        )
        if has_guided_choices and resolved_token is None:
            outcome = CommandOutcome()
            outcome.add_message("Model setup cancelled.", channel="warning", right_info="model")
            return outcome

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
            return await _pick_or_prompt_alias_token(
                io,
                items=common_items,
            )
        return None

    if isinstance(io, TuiCommandIO):
        return await _pick_or_prompt_alias_token(
            io,
            items=_merge_setup_items(items, common_items),
        )

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
        return await _prompt_manual_alias_token(io)
    return option_labels.get(normalized_selection)


async def _pick_or_prompt_alias_token(
    io: TuiCommandIO,
    *,
    items: tuple[ModelAliasSetupItem, ...],
) -> str | None:
    picker_items = tuple(
        ModelAliasPickerItem(
            token=item.token,
            priority=item.priority,
            status=f"{item.priority}/{item.status}",
            summary=item.summary,
            current_value=item.current_value,
            references=item.references,
            removable=False,
        )
        for item in items
    )
    result = await run_model_alias_picker_async(picker_items)
    if result is None:
        return None
    if result.action == "custom":
        return await _prompt_manual_alias_token(io)
    return result.token


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
    *,
    suppressed_tokens: set[str] | None = None,
) -> tuple[ModelAliasSetupItem, ...]:
    items: list[ModelAliasSetupItem] = []
    hidden_tokens = suppressed_tokens or set()
    system_aliases = valid_aliases.get("system", {})
    if "default" not in system_aliases and "$system.default" not in hidden_tokens:
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
    if "fast" not in system_aliases and "$system.fast" not in hidden_tokens:
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


def _build_picker_items(
    diagnostics: ModelAliasSetupDiagnostics,
    *,
    suppressed_tokens: set[str] | None = None,
) -> tuple[ModelAliasPickerItem, ...]:
    items: list[ModelAliasPickerItem] = []
    seen_tokens: set[str] = set()
    hidden_tokens = suppressed_tokens or set()

    def _add_item(item: ModelAliasPickerItem) -> None:
        if item.token in hidden_tokens:
            return
        if item.token in seen_tokens:
            return
        seen_tokens.add(item.token)
        items.append(item)

    for item in diagnostics.items:
        _add_item(
            ModelAliasPickerItem(
                token=item.token,
                priority=item.priority,
                status=f"{item.priority}/{item.status}",
                summary=item.summary,
                current_value=item.current_value,
                references=item.references,
                removable=False,
            )
        )

    for item in _build_common_setup_items(
        diagnostics.valid_aliases,
        suppressed_tokens=hidden_tokens,
    ):
        _add_item(
            ModelAliasPickerItem(
                token=item.token,
                priority=item.priority,
                status=f"{item.priority}/{item.status}",
                summary=item.summary,
                current_value=item.current_value,
                references=item.references,
                removable=False,
            )
        )

    for namespace, entries in sorted(diagnostics.valid_aliases.items()):
        for key, model_spec in sorted(entries.items()):
            token = f"${namespace}.{key}"
            _add_item(
                ModelAliasPickerItem(
                    token=token,
                    priority="configured",
                    status="configured",
                    summary="Existing alias mapping.",
                    current_value=model_spec,
                    references=(),
                    removable=True,
                )
            )

    return tuple(items)


async def _run_model_alias_unset(
    *,
    io: CommandIO,
    settings: Settings,
    token: str,
    target: WriteTarget,
    dry_run: bool,
) -> CommandOutcome:
    provider = _CliModelAgentProvider()
    ctx = CommandContext(
        agent_provider=provider,
        current_agent_name="cli",
        io=io,
        settings=settings,
    )
    argument = f"unset {shlex.quote(token)} --target {target}"
    if dry_run:
        argument += " --dry-run"
    return await models_manager.handle_models_command(
        ctx,
        agent_name="cli",
        action="aliases",
        argument=argument,
    )


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
    if token is not None:
        outcome = await run_model_setup(
            io=io,
            settings=settings,
            token=token,
            target=target,
            dry_run=dry_run,
        )
        for message in outcome.messages:
            await io.emit(message)
        return

    suppressed_tokens: set[str] = set()
    while True:
        diagnostics = collect_model_alias_setup_diagnostics(
            cwd=Path.cwd(),
            env_dir=getattr(settings, "environment_dir", None),
        )
        picker_items = _build_picker_items(
            diagnostics,
            suppressed_tokens=suppressed_tokens,
        )
        picker_result = await run_model_alias_picker_async(picker_items)
        if picker_result is None:
            return

        if picker_result.action == "custom":
            selected_token = await _prompt_manual_alias_token(io)
            if selected_token is None:
                return
            outcome = await run_model_setup(
                io=io,
                settings=settings,
                token=selected_token,
                target=target,
                dry_run=dry_run,
            )
        elif picker_result.action == "unset":
            assert picker_result.token is not None
            outcome = await _run_model_alias_unset(
                io=io,
                settings=settings,
                token=picker_result.token,
                target=target,
                dry_run=dry_run,
            )
            if dry_run:
                for message in outcome.messages:
                    await io.emit(message)
                return

            suppressed_tokens.add(picker_result.token)
            for message in outcome.messages:
                if message.channel in {"warning", "error"}:
                    await io.emit(message)
            continue
        else:
            assert picker_result.token is not None
            suppressed_tokens.discard(picker_result.token)
            outcome = await run_model_setup(
                io=io,
                settings=settings,
                token=picker_result.token,
                target=target,
                dry_run=dry_run,
            )

        for message in outcome.messages:
            await io.emit(message)
        if dry_run:
            return


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
