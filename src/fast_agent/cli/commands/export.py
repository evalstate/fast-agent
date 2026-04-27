"""Export persisted session traces from the CLI."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from fast_agent.cli.command_support import ensure_context_object
from fast_agent.cli.env_helpers import resolve_environment_dir_option
from fast_agent.commands.context import (
    CommandContext,
    NonInteractiveCommandIOBase,
    StaticAgentProvider,
)
from fast_agent.commands.handlers import session_export as session_export_handlers
from fast_agent.commands.handlers import sessions as session_handlers
from fast_agent.commands.session_export_help import (
    SESSION_EXPORT_AGENT_HELP,
    SESSION_EXPORT_HF_DATASET_HELP,
    SESSION_EXPORT_HF_DATASET_PATH_HELP,
    SESSION_EXPORT_OUTPUT_HELP,
    SESSION_EXPORT_PRIVACY_DEVICE_HELP,
    SESSION_EXPORT_PRIVACY_DOWNLOAD_HELP,
    SESSION_EXPORT_PRIVACY_FILTER_HELP,
    SESSION_EXPORT_PRIVACY_PATH_HELP,
    SESSION_EXPORT_PRIVACY_VARIANT_HELP,
    SESSION_EXPORT_SHOW_REDACTIONS_HELP,
    SESSION_EXPORT_TARGET_HELP,
)

if TYPE_CHECKING:
    from fast_agent.commands.results import CommandMessage


class _ExportCommandIO(NonInteractiveCommandIOBase):
    async def emit(self, message: "CommandMessage") -> None:
        del message


app = typer.Typer(
    help="Export persisted session traces.",
    context_settings={"allow_interspersed_args": True},
    add_completion=False,
)


def _render_outcome(outcome) -> None:
    for message in outcome.messages:
        text = str(message.text)
        if message.channel in {"error", "warning"}:
            typer.echo(text, err=True)
        else:
            typer.echo(text)
    if any(message.channel == "error" for message in outcome.messages):
        raise typer.Exit(1)


@app.callback(invoke_without_command=True, no_args_is_help=False)
def export(
    ctx: typer.Context,
    target: str | None = typer.Argument(
        None,
        help=SESSION_EXPORT_TARGET_HELP,
    ),
    list_sessions: bool = typer.Option(
        False,
        "--list",
        help="List recent sessions instead of exporting.",
    ),
    agent: str | None = typer.Option(None, "--agent", "-a", help=SESSION_EXPORT_AGENT_HELP),
    output: Path | None = typer.Option(None, "--output", "-o", help=SESSION_EXPORT_OUTPUT_HELP),
    hf_dataset: str | None = typer.Option(
        None,
        "--hf-dataset",
        help=SESSION_EXPORT_HF_DATASET_HELP,
    ),
    hf_dataset_path: str | None = typer.Option(
        None,
        "--hf-dataset-path",
        help=SESSION_EXPORT_HF_DATASET_PATH_HELP,
    ),
    privacy_filter: bool = typer.Option(
        False,
        "--privacy-filter",
        help=SESSION_EXPORT_PRIVACY_FILTER_HELP,
    ),
    privacy_filter_path: Path | None = typer.Option(
        None,
        "--privacy-filter-path",
        help=SESSION_EXPORT_PRIVACY_PATH_HELP,
    ),
    download_privacy_filter: bool = typer.Option(
        False,
        "--download-privacy-filter",
        help=SESSION_EXPORT_PRIVACY_DOWNLOAD_HELP,
    ),
    privacy_filter_device: str | None = typer.Option(
        None,
        "--privacy-filter-device",
        help=SESSION_EXPORT_PRIVACY_DEVICE_HELP,
    ),
    privacy_filter_variant: str | None = typer.Option(
        None,
        "--privacy-filter-variant",
        "--privacy-filter-quant",
        help=SESSION_EXPORT_PRIVACY_VARIANT_HELP,
    ),
    show_redactions: bool = typer.Option(
        False,
        "--show-redactions",
        help=SESSION_EXPORT_SHOW_REDACTIONS_HELP,
    ),
) -> None:
    """Export a persisted session trace."""
    context_payload = ensure_context_object(ctx)
    env_dir_value = context_payload.get("env_dir")
    env_dir = env_dir_value if isinstance(env_dir_value, Path) else None
    resolve_environment_dir_option(ctx, env_dir)

    command_context = CommandContext(
        agent_provider=StaticAgentProvider(),
        current_agent_name="cli",
        io=_ExportCommandIO(),
    )
    if list_sessions:
        if (
            target is not None
            or agent is not None
            or output is not None
            or hf_dataset is not None
            or hf_dataset_path is not None
            or privacy_filter
            or privacy_filter_path is not None
            or download_privacy_filter
            or privacy_filter_device is not None
            or privacy_filter_variant is not None
            or show_redactions
        ):
            raise typer.BadParameter("Cannot combine --list with export options.")
        outcome = asyncio.run(
            session_handlers.handle_list_sessions(
                command_context,
                show_help=False,
            )
        )
        _render_outcome(outcome)
        return

    outcome = asyncio.run(
        session_export_handlers.handle_session_export(
            command_context,
            target=target,
            agent_name=agent,
            output_path=str(output) if output is not None else None,
            hf_dataset=hf_dataset,
            hf_dataset_path=hf_dataset_path,
            privacy_filter=privacy_filter,
            privacy_filter_path=(
                str(privacy_filter_path) if privacy_filter_path is not None else None
            ),
            download_privacy_filter=download_privacy_filter,
            privacy_filter_device=privacy_filter_device,
            privacy_filter_variant=privacy_filter_variant,
            show_redactions=show_redactions,
            progress_callback=lambda message: typer.echo(message, err=True),
        )
    )
    _render_outcome(outcome)
