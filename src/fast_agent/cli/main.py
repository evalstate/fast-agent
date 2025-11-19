"""Main CLI entry point for MCP Agent."""

import typer
from rich.table import Table

from fast_agent.cli.commands import acp, auth, check_config, go, quickstart, serve, setup
from fast_agent.cli.terminal import Application
from fast_agent.ui.console import console as shared_console

app = typer.Typer(
    help="Use `fast-agent go --help` for interactive shell arguments and options.",
    add_completion=False,  # We'll add this later when we have more commands
)

# Subcommands
app.add_typer(go.app, name="go", help="Run an interactive agent directly from the command line")
app.add_typer(serve.app, name="serve", help="Run FastAgent as an MCP server")
app.add_typer(acp.app, name="acp", help="Run FastAgent as an ACP stdio server")
app.add_typer(setup.app, name="setup", help="Set up a new agent project")
app.add_typer(check_config.app, name="check", help="Show or diagnose fast-agent configuration")
app.add_typer(auth.app, name="auth", help="Manage OAuth authentication for MCP servers")
app.add_typer(quickstart.app, name="bootstrap", help="Create example applications")
app.add_typer(quickstart.app, name="quickstart", help="Create example applications")

# Shared application context
application = Application()
# Use shared console to match app-wide styling
console = shared_console


def show_welcome() -> None:
    """Show a welcome message with available commands, using new styling."""
    from importlib.metadata import version

    from rich.text import Text

    try:
        app_version = version("fast-agent-mcp")
    except:  # noqa: E722
        app_version = "unknown"

    # Header in the same style used by check/console_display
    def _print_section_header(title: str, color: str = "blue") -> None:
        width = console.size.width
        left = f"[{color}]▎[/{color}][dim {color}]▶[/dim {color}] [{color}]{title}[/{color}]"
        left_text = Text.from_markup(left)
        separator_count = max(1, width - left_text.cell_len - 1)

        combined = Text()
        combined.append_text(left_text)
        combined.append(" ")
        combined.append("─" * separator_count, style="dim")

        console.print()
        console.print(combined)
        console.print()

    header_title = f"fast-agent v{app_version}"
    _print_section_header(header_title, color="blue")

    # Commands list (no boxes), matching updated check styling
    table = Table(show_header=True, box=None)
    table.add_column("Command", style="green", header_style="bold bright_white")
    table.add_column("Description", header_style="bold bright_white")

    table.add_row("[bold]go[/bold]", "Start an interactive session")
    table.add_row("go -x", "Start an interactive session with a local shell tool")
    table.add_row("[bold]serve[/bold]", "Start fast-agent as an MCP server")
    table.add_row("check", "Show current configuration")
    table.add_row("auth", "Manage OAuth tokens and keyring")
    table.add_row("setup", "Create agent template and configuration")
    table.add_row("quickstart", "Create example applications (workflow, researcher, etc.)")

    console.print(table)

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
) -> None:
    """fast-agent - Build effective agents using Model Context Protocol (MCP).

    Use --help with any command for detailed usage information.
    """
    application.verbosity = 1 if verbose else 0 if not quiet else -1
    if not color:
        # Recreate consoles without color when --no-color is provided
        from fast_agent.ui.console import console as base_console
        from fast_agent.ui.console import error_console as base_error_console

        application.console = base_console.__class__(color_system=None)
        application.error_console = base_error_console.__class__(color_system=None, stderr=True)

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
        show_welcome()
