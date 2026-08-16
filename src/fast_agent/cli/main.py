"""Main CLI entry point for MCP Agent."""

import importlib
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import ClassVar

import click
import typer
import typer.main
from typer.core import TyperGroup

from fast_agent.cli.command_support import ensure_context_object
from fast_agent.cli.constants import normalize_convenience_flag_args, normalize_resume_flag_args
from fast_agent.cli.display import print_section_header
from fast_agent.cli.home_helpers import resolve_workspace_and_home_options
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
    "session": "fast_agent.cli.commands.session:app",
    "batch": "fast_agent.cli.commands.batch:app",
    "quickstart": "fast_agent.cli.commands.quickstart:app",
    "bootstrap": "fast_agent.cli.commands.quickstart:app",
    "demo": "fast_agent.cli.commands.demo:app",
    "export": "fast_agent.cli.commands.export:app",
}

LAZY_SUBCOMMAND_HELP: dict[str, str] = {
    "acp": "Start fast-agent as an ACP stdio server (convenience wrapper for 'serve --transport acp').",
    "auth": "Inspect and manage provider and MCP credentials.",
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
    "session": "Maintain persisted sessions.",
    "skills": (
        "Manage skills (list/available/search/add/remove/update). Add supports marketplace "
        "selectors, GitHub URLs, GitHub paths, and local paths."
    ),
}

_ROOT_HELP_CONTEXT_KEY = "fast_agent_lazy_root_help"


def _resolve_root_verbosity(*, verbose: bool, quiet: bool) -> int:
    if verbose:
        return 1
    if quiet:
        return -1
    return 0


def _installed_package_version(package_name: str) -> str:
    try:
        return package_version(package_name)
    except PackageNotFoundError:
        return "unknown"


class LazyGroup(TyperGroup):
    lazy_subcommands: ClassVar[dict[str, str]] = {}
    lazy_subcommand_help: ClassVar[dict[str, str]] = {}

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        normalize_convenience_flag_args(args)
        if _first_root_command(args) == "go":
            normalize_resume_flag_args(args)
        return super().parse_args(ctx, args)

    def list_commands(self, ctx: click.Context) -> list[str]:
        del ctx
        return sorted(self.lazy_subcommands)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        if ctx.meta.get(_ROOT_HELP_CONTEXT_KEY) is True:
            short_help = self.lazy_subcommand_help.get(cmd_name)
            return click.Command(name=cmd_name, help=short_help) if short_help is not None else None
        target = self.lazy_subcommands.get(cmd_name)
        if not target:
            return None
        module_path, app_name = target.split(":", 1)
        module = importlib.import_module(module_path)
        typer_app = getattr(module, app_name)
        command = typer.main.get_command(typer_app)
        command.name = cmd_name
        return command

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        rows = [
            (command_name, self.lazy_subcommand_help[command_name])
            for command_name in self.list_commands(ctx)
        ]
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)

    def format_help(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        ctx.meta[_ROOT_HELP_CONTEXT_KEY] = True
        try:
            super().format_help(ctx, formatter)
        finally:
            ctx.meta.pop(_ROOT_HELP_CONTEXT_KEY, None)


app = typer.Typer(
    cls=LazyGroup,
    help="Use `fast-agent go --help` for interactive shell arguments and options.",
    add_completion=False,  # We'll add this later when we have more commands
)
LazyGroup.lazy_subcommands = LAZY_SUBCOMMANDS
LazyGroup.lazy_subcommand_help = LAZY_SUBCOMMAND_HELP


def _first_root_command(args: list[str]) -> str | None:
    index = 0
    while index < len(args):
        arg = args[index]
        if arg == "--":
            return args[index + 1] if index + 1 < len(args) else None
        if arg in {"--home"}:
            index += 2
            continue
        if arg.startswith("--home="):
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
    from rich.table import Table

    header_title = f"fast-agent v{_installed_package_version('fast-agent-mcp')}"
    print_section_header(console, header_title, color="blue")

    # Commands list (no boxes), matching updated check styling
    table = Table(show_header=True, box=None)
    table.add_column("Command", style="green", header_style="bold bright_white")
    table.add_column("Description", header_style="bold bright_white")

    table.add_row("[bold]go[/bold]", "Start an interactive session")
    table.add_row("go -x", "Start an interactive session with a local shell tool")
    table.add_row("[bold]serve[/bold]", "Expose fast-agent over MCP (http/stdio) or ACP")
    table.add_row(
        "[bold]acp[/bold]", "Start fast-agent as an ACP stdio server (for Zed, Toad, etc.)"
    )
    table.add_row("[bold]export[/bold]", "Export a persisted session trace")
    table.add_row("session", "Maintain persisted sessions")
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
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        help="Override the workspace root; default home resolves under this directory",
    ),
    home: Path | None = typer.Option(
        None,
        "--home",
        help="Use PATH as the fast-agent home itself (contains agent-cards/, sessions/, etc.)",
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

    resolved_workspace, resolved_home = resolve_workspace_and_home_options(
        ctx,
        workspace=workspace,
        home=home,
    )
    context_payload["workspace"] = resolved_workspace
    context_payload["home"] = resolved_home

    application.verbosity = _resolve_root_verbosity(verbose=verbose, quiet=quiet)
    if not color:
        # Recreate consoles without color when --no-color is provided
        from fast_agent.ui.console import console as base_console
        from fast_agent.ui.console import error_console as base_error_console

        application.console = base_console.__class__(color_system=None)
        application.error_console = base_error_console.__class__(color_system=None, stderr=True)

    update_notice: str | None = None
    if (
        not version
        and ctx.invoked_subcommand is None
        and should_run_update_check(
            disabled=no_update_check,
        )
    ):
        update_notice = check_for_update_notice(home=resolved_home)

    # Handle version flag
    if version:
        console.print(f"fast-agent-mcp v{_installed_package_version('fast-agent-mcp')}")
        raise typer.Exit()

    # Show welcome message if no command was invoked
    if ctx.invoked_subcommand is None:
        show_welcome(update_notice=update_notice)
