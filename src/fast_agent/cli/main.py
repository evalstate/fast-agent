"""Main CLI entry point for MCP Agent."""

from __future__ import annotations

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


def _get_available_providers() -> set[str]:
    """Detect which providers have API keys configured (env vars or config file).

    Returns a set of provider names (lowercase) that have valid keys.
    """
    from pathlib import Path

    from fast_agent.cli.commands.check_config import find_config_files, get_secrets_summary
    from fast_agent.llm.provider_key_manager import API_KEY_HINT_TEXT, ProviderKeyManager
    from fast_agent.llm.provider_types import Provider

    available: set[str] = set()

    # Find config files from current directory
    config_files = find_config_files(Path.cwd())
    secrets_summary = get_secrets_summary(config_files.get("secrets"))

    secrets = secrets_summary.get("secrets", {})
    secrets_status = secrets_summary.get("status", "not_found")

    for provider in Provider:
        if provider == Provider.FAST_AGENT:
            available.add(provider.value)
            continue

        provider_name = provider.value

        # Check environment variable
        env_key = ProviderKeyManager.get_env_var(provider_name)
        if env_key:
            available.add(provider_name)
            continue

        # Check secrets file
        if secrets_status == "parsed":
            config_key = ProviderKeyManager.get_config_file_key(provider_name, secrets)
            if config_key and config_key != API_KEY_HINT_TEXT:
                available.add(provider_name)
                continue

    return available


def _get_aliases_by_provider() -> dict[str, list[tuple[str, str]]]:
    """Group model aliases by their resolved provider.

    Returns a dict mapping provider name to list of (alias, resolved_model) tuples.
    """
    from fast_agent.llm.model_factory import ModelFactory

    aliases_by_provider: dict[str, list[tuple[str, str]]] = {}

    for alias, resolved in ModelFactory.MODEL_ALIASES.items():
        try:
            config = ModelFactory.parse_model_string(resolved)
            provider_name = config.provider.value
        except Exception:
            provider_name = "unknown"

        if provider_name not in aliases_by_provider:
            aliases_by_provider[provider_name] = []
        aliases_by_provider[provider_name].append((alias, config.model_name if 'config' in dir() else resolved))

    # Re-parse to get proper model names
    for alias, resolved in ModelFactory.MODEL_ALIASES.items():
        try:
            config = ModelFactory.parse_model_string(resolved)
            provider_name = config.provider.value
            # Update the entry with proper model name
            for i, (a, _) in enumerate(aliases_by_provider.get(provider_name, [])):
                if a == alias:
                    aliases_by_provider[provider_name][i] = (alias, config.model_name)
                    break
        except Exception:
            pass

    return aliases_by_provider


# Curated list of commonly used aliases to show in welcome message
# Format: (alias, description/model hint)
FEATURED_ALIASES: list[tuple[str, str, str]] = [
    # (alias, provider, description)
    ("sonnet", "anthropic", "Claude Sonnet 4.5"),
    ("haiku", "anthropic", "Claude Haiku 4.5"),
    ("opus", "anthropic", "Claude Opus 4.5"),
    ("gpt51", "openai", "GPT 5.1"),
    ("o3", "openai", "o3"),
    ("gemini25", "google", "Gemini 2.5 Flash"),
    ("gemini25pro", "google", "Gemini 2.5 Pro"),
    ("deepseek", "deepseek", "DeepSeek Chat"),
    ("kimi", "hf", "Kimi K2 (via Groq)"),
    ("qwen3", "hf", "Qwen3 (via Together)"),
]


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

    # Show model aliases with availability indicators
    _print_section_header("Model Aliases", color="blue")

    available_providers = _get_available_providers()

    # Build alias display table
    alias_table = Table(show_header=True, box=None, padding=(0, 2, 0, 0))
    alias_table.add_column("", style="dim", width=2)  # Status indicator
    alias_table.add_column("Alias", style="cyan", header_style="bold bright_white")
    alias_table.add_column("Model", header_style="bold bright_white")
    alias_table.add_column("Provider", style="dim", header_style="bold bright_white")

    for alias, provider, description in FEATURED_ALIASES:
        is_available = provider in available_providers
        status = "[green]✓[/green]" if is_available else "[dim]✗[/dim]"
        alias_style = "[bold cyan]" if is_available else "[dim]"
        desc_style = "" if is_available else "[dim]"
        provider_style = "[green]" if is_available else "[dim]"

        alias_table.add_row(
            status,
            f"{alias_style}{alias}[/]",
            f"{desc_style}{description}[/]",
            f"{provider_style}{provider}[/]",
        )

    console.print(alias_table)
    console.print(
        "\n[dim]✓ = API key configured  |  Use[/dim] [cyan]fast-agent check[/cyan] [dim]for full status[/dim]"
    )

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
