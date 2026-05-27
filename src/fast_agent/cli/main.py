"""Main CLI entry point for MCP Agent."""

import importlib
import os
from pathlib import Path

import click
import typer
import typer.main
from typer.core import TyperGroup

from fast_agent.cli.command_support import ensure_context_object
from fast_agent.cli.constants import normalize_resume_flag_args
from fast_agent.cli.display import print_section_header
from fast_agent.cli.env_helpers import resolve_environment_dir_option
from fast_agent.cli.terminal import Application
from fast_agent.cli.update_check import check_for_update_notice, should_run_update_check
from fast_agent.constants import FAST_AGENT_SHELL_CHILD_ENV
from fast_agent.ui.console import console as shared_console

LAZY_SUBCOMMANDS: dict[str, str] = {
    "go": "fast_agent.cli.commands.go:app",
    "serve": "fast_agent.cli.commands.serve:app",
    "acp": "fast_agent.cli.commands.acp:app",
    "scaffold": "fast_agent.cli.commands.setup:app",
    "check": "fast_agent.cli.commands.check_config:app",
    "cards": "fast_agent.cli.commands.cards:app",
    "plugins": "fast_agent.cli.commands.plugins:app",
    "skills": "fast_agent.cli.commands.skills:app",
    "config": "fast_agent.cli.commands.config:app",
    "model": "fast_agent.cli.commands.model:app",
    "auth": "fast_agent.cli.commands.auth:app",
    "batch": "fast_agent.cli.commands.batch:app",
    "quickstart": "fast_agent.cli.commands.quickstart:app",
    "bootstrap": "fast_agent.cli.commands.quickstart:app",
    "demo": "fast_agent.cli.commands.demo:app",
    "export": "fast_agent.cli.commands.export:app",
}

LAZY_SUBCOMMAND_HELP: dict[str, str] = {
    "acp": "Start fast-agent as an ACP stdio server (convenience wrapper for 'serve --transport acp').",
    "auth": "Manage OAuth tokens stored in the OS keyring for MCP HTTP/SSE servers (identity = base URL).",
    "batch": "Run batch processing jobs.",
    "bootstrap": "Create fast-agent quickstarts",
    "cards": "Manage card packs (list/add/remove/update/publish).",
    "check": "Check and diagnose FastAgent configuration",
    "config": "Configure fast-agent settings interactively.",
    "demo": "Demo commands for UI features.",
    "export": "Export persisted session traces.",
    "go": "Run an interactive agent directly from the command line without creating an agent.py file",
    "model": "Interactive model reference setup.",
    "plugins": "Manage command plugins (list/add/remove/update).",
    "quickstart": "Create fast-agent quickstarts",
    "scaffold": "Initialize a new FastAgent project with configuration files and example agent.",
    "serve": "Expose fast-agent to clients over MCP (http or stdio), ACP, or A2A, without writing an agent.py file",
    "skills": "Manage skills (list/available/search/add/remove/update).",
}


class LazyGroup(TyperGroup):
    lazy_subcommands: dict[str, str] = {}
    lazy_help: dict[str, str] = {}
    _rendering_root_help: bool = False

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if _first_root_command(args) == "go":
            normalize_resume_flag_args(args)
        return super().parse_args(ctx, args)

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(self.lazy_subcommands)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        target = self.lazy_subcommands.get(cmd_name)
        if not target:
            return None
        if self._rendering_root_help:
            return click.Command(cmd_name, help=self.lazy_help.get(cmd_name, ""))
        module_path, app_name = target.split(":", 1)
        module = importlib.import_module(module_path)
        typer_app = getattr(module, app_name)
        command = typer.main.get_command(typer_app)
        command.name = cmd_name
        return command

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        self._rendering_root_help = True
        try:
            super().format_help(ctx, formatter)
        finally:
            self._rendering_root_help = False


app = typer.Typer(
    cls=LazyGroup,
    help="Use `fast-agent go --help` for interactive shell arguments and options.",
    add_completion=False,  # We'll add this later when we have more commands
)
LazyGroup.lazy_subcommands = LAZY_SUBCOMMANDS
LazyGroup.lazy_help = LAZY_SUBCOMMAND_HELP


def _first_root_command(args: list[str]) -> str | None:
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            return args[index + 1] if index + 1 < len(args) else None
        if arg in {"--env"}:
            index += 2
            continue
        if arg.startswith("--env="):
            index += 1
            continue
        if arg.startswith("-") and arg != "-":
            index += 1
            continue
        return arg
    return None

# Shared application context
application = Application()
# Use shared console to match app-wide styling
console = shared_console


def show_welcome(update_notice: str | None = None) -> None:
    """Show a welcome message with available commands, using new styling."""
    from importlib.metadata import version

    from rich.table import Table

    try:
        app_version = version("fast-agent-mcp")
    except:  # noqa: E722
        app_version = "unknown"

    header_title = f"fast-agent v{app_version}"
    print_section_header(console, header_title, color="blue")

    # Commands list (no boxes), matching updated check styling
    table = Table(show_header=True, box=None)
    table.add_column("Command", style="green", header_style="bold bright_white")
    table.add_column("Description", header_style="bold bright_white")

    table.add_row("[bold]go[/bold]", "Start an interactive session")
    table.add_row("go -x", "Start an interactive session with a local shell tool")
    table.add_row("[bold]serve[/bold]", "Expose fast-agent over MCP (http/stdio) or ACP")
    table.add_row("[bold]acp[/bold]", "Start fast-agent as an ACP stdio server (for Zed, Toad, etc.)")
    table.add_row("[bold]export[/bold]", "Export a persisted session trace")
    table.add_row("check", "Show current configuration")
    table.add_row("cards", "Manage card packs (list/add/remove/update/publish)")
    table.add_row("plugins", "Manage command plugins (list/add/remove/update)")
    table.add_row("skills", "Manage skills (list/available/search/add/remove/update)")
    table.add_row("config", "Configure settings interactively (shell, model)")
    table.add_row("auth", "Manage OAuth tokens in the OS keyring for MCP servers")
    table.add_row("batch", "Run batch processing jobs")
    table.add_row("scaffold", "Create agent template and configuration")
    table.add_row("quickstart", "Create example applications (workflow, researcher, etc.)")
    table.add_row("demo", "Run local UI demos (no model calls)")

    console.print(table)

    if update_notice:
        console.print()
        console.print(update_notice)

    console.print(
        "\nVisit [cyan][link=https://fast-agent.ai]fast-agent.ai[/link][/cyan] for more information."
    )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose mode"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Disable output"),
    color: bool = typer.Option(True, "--color/--no-color", help="Enable/disable color output"),
    version: bool = typer.Option(False, "--version", help="Show version and exit"),
    no_update_check: bool = typer.Option(
        False,
        "--no-update-check",
        help="Skip checking PyPI for newer fast-agent releases",
    ),
    env: Path | None = typer.Option(
        None, "--env", help="Override the base fast-agent environment directory"
    ),
) -> None:
    """fast-agent - Build effective agents using Model Context Protocol (MCP).

    Use --help with any command for detailed usage information.
    """
    if os.getenv(FAST_AGENT_SHELL_CHILD_ENV):
        typer.echo(
            "fast-agent is already running inside a fast-agent shell command. "
            "Exit the shell or unset FAST_AGENT_SHELL_CHILD to continue.",
            err=True,
        )
        raise typer.Exit(1)

    context_payload = ensure_context_object(ctx)
    context_payload["no_update_check"] = no_update_check

    resolved_env_dir = resolve_environment_dir_option(ctx, env)
    context_payload["env_dir"] = resolved_env_dir

    application.verbosity = 1 if verbose else 0 if not quiet else -1
    if not color:
        # Recreate consoles without color when --no-color is provided
        from fast_agent.ui.console import console as base_console
        from fast_agent.ui.console import error_console as base_error_console

        application.console = base_console.__class__(color_system=None)
        application.error_console = base_error_console.__class__(color_system=None, stderr=True)

    update_notice: str | None = None
    if not version and ctx.invoked_subcommand is None and should_run_update_check(
        disabled=no_update_check,
    ):
        update_notice = check_for_update_notice(environment_dir=resolved_env_dir)

    # Handle version flag
    if version:
        from importlib.metadata import version as get_version

        try:
            app_version = get_version("fast-agent-mcp")
        except:  # noqa: E722
            app_version = "unknown"
        console.print(f"fast-agent-mcp v{app_version}")
        raise typer.Exit()

    # Show welcome message if no command was invoked
    if ctx.invoked_subcommand is None:
        show_welcome(update_notice=update_notice)
