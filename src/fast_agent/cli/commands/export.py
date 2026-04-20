"""Export persisted session traces from the CLI."""

from __future__ import annotations

import asyncio
from pathlib import Path

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


class _ExportCommandIO(NonInteractiveCommandIOBase):
    async def emit(self, message) -> None:
        del message


app = typer.Typer(
    help="Export persisted session traces.",
    context_settings={"allow_interspersed_args": True},
    add_completion=False,
)


def _render_outcome(outcome) -> None:
    has_error = False
    for message in outcome.messages:
        text = str(message.text)
        if message.channel == "error":
            has_error = True
            typer.echo(text, err=True)
        else:
            typer.echo(text)
    if has_error:
        raise typer.Exit(1)


@app.callback(invoke_without_command=True, no_args_is_help=False)
def export(
    ctx: typer.Context,
    target: str | None = typer.Argument(
        None,
        help="Session target: latest, session id, session dir, or session.json path.",
    ),
    list_sessions: bool = typer.Option(
        False,
        "--list",
        help="List recent sessions instead of exporting.",
    ),
    agent: str | None = typer.Option(None, "--agent", "-a", help="Agent name to export."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write trace to this path."),
    hf_dataset: str | None = typer.Option(
        None,
        "--hf-dataset",
        help="Upload the exported trace to this Hugging Face dataset repo (owner/name).",
    ),
    hf_dataset_path: str | None = typer.Option(
        None,
        "--hf-dataset-path",
        help=(
            "Path in the dataset repo. Defaults to the root using the local filename. "
            "If the value ends with '/', it is treated as a folder."
        ),
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
        )
    )
    _render_outcome(outcome)
