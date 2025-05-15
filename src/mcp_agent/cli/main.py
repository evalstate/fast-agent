"""Main CLI entry point for MCP Agent."""

import typer
from rich.console import Console
from rich.table import Table

from mcp_agent.cli.commands import check_config, go, quickstart, setup
from mcp_agent.cli.terminal import Application

app = typer.Typer(
    help="FastAgent CLI - Build effective agents using Model Context Protocol",
    add_completion=False,  # We'll add this later when we have more commands
)

# Subcommands
app.add_typer(go.app, name="go", help="Run an interactive agent directly from the command line")
app.add_typer(setup.app, name="setup", help="Set up a new agent project")
app.add_typer(check_config.app, name="check", help="Show or diagnose fast-agent configuration")
app.add_typer(quickstart.app, name="bootstrap", help="Create example applications")
app.add_typer(quickstart.app, name="quickstart", help="Create example applications")

# Shared application context
application = Application()
console = Console()


def show_welcome() -> None:
    """Show a welcome message with available commands."""
    from importlib.metadata import version

    try:
        app_version = version("fast-agent-mcp")
    except:  # noqa: E722
        app_version = "unknown"

    console.print(f"\nfast-agent {app_version} [dim](fast-agent-mcp)[/dim] ")

    # Create a table for commands
    table = Table(title="\nAvailable Commands")
    table.add_column("Command", style="green")
    table.add_column("Description")

    table.add_row("[bold]go[/bold]", "Start an interactive session with an agent")
    table.add_row("setup", "Create a new agent template and configuration files")
    table.add_row("check", "Show or diagnose fast-agent configuration")
    table.add_row("quickstart", "Create example applications (workflow, researcher, etc.)")

    console.print(table)

    console.print(
        "\n[italic]get started with:[/italic] [bold][cyan]fast-agent[/cyan][/bold] [green]setup[/green]"
    )


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose mode"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Disable output"),
    color: bool = typer.Option(True, "--color/--no-color", help="Enable/disable color output"),
    version: bool = typer.Option(False, "--version", help="Show version and exit"),
) -> None:
    """FastAgent CLI - Build effective agents using Model Context Protocol (MCP).

    Use --help with any command for detailed usage information.
    """
    application.verbosity = 1 if verbose else 0 if not quiet else -1
    application.console = application.console if color else None

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
