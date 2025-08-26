"""Bootstrap command to create example applications."""

import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    help="Create fast-agent quickstarts",
    no_args_is_help=False,  # Allow showing our custom help instead
)
console = Console()

EXAMPLE_TYPES = {
    "workflow": {
        "description": "Example workflows, demonstrating each of the patterns in Anthropic's\n"
        "'Building Effective Agents' paper. Some agents use the 'fetch'\n"
        "and filesystem MCP Servers.",
        "files": [
            "chaining.py",
            "evaluator.py",
            "human_input.py",
            "orchestrator.py",
            "parallel.py",
            "router.py",
            "short_story.txt",
            "fastagent.config.yaml",
        ],
        "create_subdir": True,
    },
    "researcher": {
        "description": "Research agent example with additional evaluation/optimization\n"
        "example. Uses Brave Search and Docker MCP Servers.\n"
        "Creates examples in a 'researcher' subdirectory.",
        "files": ["researcher.py", "researcher-eval.py", "fastagent.config.yaml"],
        "create_subdir": True,
    },
    "data-analysis": {
        "description": "Data analysis agent examples that demonstrate working with\n"
        "datasets, performing statistical analysis, and generating visualizations.\n"
        "Creates examples in a 'data-analysis' subdirectory with mount-point for data.\n"
        "Uses MCP 'roots' feature for mapping",
        "files": ["analysis.py", "fastagent.config.yaml"],
        "mount_point_files": ["WA_Fn-UseC_-HR-Employee-Attrition.csv"],
        "create_subdir": True,
    },
    "state-transfer": {
        "description": "Example demonstrating state transfer between multiple agents.\n"
        "Shows how state can be passed between agent runs to maintain context.\n"
        "Creates examples in a 'state-transfer' subdirectory.",
        "files": [
            "agent_one.py",
            "agent_two.py",
            "fastagent.config.yaml",
            "fastagent.secrets.yaml.example",
        ],
        "create_subdir": True,
    },
    "elicitations": {
        "description": "Interactive form examples using MCP elicitations feature.\n"
        "Demonstrates collecting structured data with forms, AI-guided workflows,\n"
        "and custom handlers. Creates examples in an 'elicitations' subdirectory.",
        "files": [
            "elicitation_account_server.py",
            "elicitation_forms_server.py",
            "elicitation_game_server.py",
            "fastagent.config.yaml",
            "fastagent.secrets.yaml.example",
            "forms_demo.py",
            "game_character.py",
            "game_character_handler.py",
            "tool_call.py",
        ],
        "create_subdir": True,
    },
    "tensorzero": {
        "description": "A complete example showcasing the TensorZero integration.\n"
        "Includes the T0 Gateway, an MCP server, an interactive agent, and \n"
        "multi-modal functionality.",
        "files": [
            ".env.sample",
            "Makefile",
            "README.md",
            "agent.py",
            "docker-compose.yml",
            "fastagent.config.yaml",
            "image_demo.py",
            "simple_agent.py",
            "mcp_server/",
            "demo_images/",
            "tensorzero_config/"
        ],
        "create_subdir": True,
    },
}


def copy_example_files(example_type: str, target_dir: Path, force: bool = False) -> list[str]:
    """Copy example files from resources to target directory."""
    created = []

    # Determine if we should create a subdirectory for this example type
    example_info = EXAMPLE_TYPES[example_type]
    if example_info["create_subdir"]:
        target_dir = target_dir / example_type
        if not target_dir.exists():
            target_dir.mkdir(parents=True)
            console.print(f"Created subdirectory: {target_dir}")

    # Create mount-point directory if needed
    mount_point_files = example_info.get("mount_point_files", [])
    if mount_point_files:
        mount_point_dir = target_dir / "mount-point"
        if not mount_point_dir.exists():
            mount_point_dir.mkdir(parents=True)
            console.print(f"Created mount-point directory: {mount_point_dir}")

    # Try to use examples from the installed package first, or fall back to the top-level directory
    from importlib.resources import files

    try:
        # First try to find examples in the package resources
        if example_type == "state-transfer":
            # The state-transfer example is in the mcp subdirectory
            source_dir = (
                files("mcp_agent")
                .joinpath("resources")
                .joinpath("examples")
                .joinpath("mcp")
                .joinpath("state-transfer")
            )
        elif example_type == "elicitations":
            # The elicitations example is in the mcp subdirectory
            source_dir = (
                files("mcp_agent")
                .joinpath("resources")
                .joinpath("examples")
                .joinpath("mcp")
                .joinpath("elicitations")
            )
        else:
            # Other examples are at the top level of examples
            source_dir = (
                files("mcp_agent")
                .joinpath("resources")
                .joinpath("examples")
                .joinpath("workflows" if example_type == "workflow" else f"{example_type}")
            )

        # Check if we found a valid directory
        if not source_dir.is_dir():
            console.print(
                f"[yellow]Resource directory not found: {source_dir}. Falling back to development mode.[/yellow]"
            )
            # Fall back to the top-level directory for development mode
            package_dir = Path(__file__).parent.parent.parent.parent.parent
            if example_type == "state-transfer":
                source_dir = package_dir / "examples" / "mcp" / "state-transfer"
            elif example_type == "elicitations":
                source_dir = package_dir / "examples" / "mcp" / "elicitations"
            else:
                source_dir = (
                    package_dir
                    / "examples"
                    / ("workflows" if example_type == "workflow" else f"{example_type}")
                )
            console.print(f"[blue]Using development directory: {source_dir}[/blue]")
    except (ImportError, ModuleNotFoundError, ValueError) as e:
        console.print(
            f"[yellow]Error accessing resources: {e}. Falling back to development mode.[/yellow]"
        )
        # Fall back to the top-level directory if the resource finding fails
        package_dir = Path(__file__).parent.parent.parent.parent.parent
        if example_type == "state-transfer":
            source_dir = package_dir / "examples" / "mcp" / "state-transfer"
        elif example_type == "elicitations":
            source_dir = package_dir / "examples" / "mcp" / "elicitations"
        else:
            source_dir = (
                package_dir
                / "examples"
                / ("workflows" if example_type == "workflow" else f"{example_type}")
            )

    if not source_dir.exists():
        console.print(f"[red]Error: Source directory not found: {source_dir}[/red]")
        return created

    for filename in example_info["files"]:
        source = source_dir / filename
        target = target_dir / filename

        try:
            if not source.exists():
                console.print(f"[red]Error: Source file not found: {source}[/red]")
                continue

            if target.exists() and not force:
                console.print(f"[yellow]Skipping[/yellow] {filename} (already exists)")
                continue

            shutil.copy2(source, target)
            try:
                # This can fail in test environments where the target is not relative to target_dir.parent
                rel_path = str(target.relative_to(target_dir.parent))
            except ValueError:
                # Fallback to just the filename
                rel_path = f"{example_type}/{filename}"

            created.append(rel_path)
            console.print(f"[green]Created[/green] {created[-1]}")

        except Exception as e:
            console.print(f"[red]Error copying {filename}: {str(e)}[/red]")

    # Copy mount-point files if any
    if mount_point_files:
        source_mount_point = source_dir / "mount-point"
        for filename in mount_point_files:
            source = source_mount_point / filename
            target = mount_point_dir / filename

            try:
                if not source.exists():
                    console.print(f"[red]Error: Source file not found: {source}[/red]")
                    continue

                if target.exists() and not force:
                    console.print(
                        f"[yellow]Skipping[/yellow] mount-point/{filename} (already exists)"
                    )
                    continue

                shutil.copy2(source, target)
                created.append(f"{example_type}/mount-point/{filename}")
                console.print(f"[green]Created[/green] mount-point/{filename}")

            except Exception as e:
                console.print(f"[red]Error copying mount-point/{filename}: {str(e)}[/red]")

    return created


def copy_project_template(source_dir: Path, dest_dir: Path, console: Console, force: bool = False):
    """
    Recursively copies a project template directory.
    This is a helper to handle project-based quickstarts like TensorZero.
    """
    if dest_dir.exists():
        if force:
            console.print(f"[yellow]--force specified. Removing existing directory: {dest_dir}[/yellow]")
            shutil.rmtree(dest_dir)
        else:
            console.print(f"[bold yellow]Directory '{dest_dir.name}' already exists.[/bold yellow] Use --force to overwrite.")
            return False

    try:
        shutil.copytree(source_dir, dest_dir)
        return True
    except Exception as e:
        console.print(f"[red]Error copying project template: {e}[/red]")
        return False


def show_overview() -> None:
    """Display an overview of available examples in a nicely formatted table."""
    console.print("\n[bold cyan]fast-agent quickstarts[/bold cyan]")
    console.print("Build agents and compose workflows through practical examples\n")

    # Create a table for better organization
    table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 2))
    table.add_column("Example")
    table.add_column("Description")
    table.add_column("Files")

    for name, info in EXAMPLE_TYPES.items():
        # Just show file count instead of listing all files
        file_count = len(info["files"])
        files_summary = f"{file_count} files"
        if "mount_point_files" in info:
            mount_count = len(info["mount_point_files"])
            files_summary += f"\n+ {mount_count} data files"
        table.add_row(f"[green]{name}[/green]", info["description"], files_summary)

    console.print(table)

    # Show usage instructions in a panel
    usage_text = (
        "[bold]Usage:[/bold]\n"
        "  [cyan]fast-agent[/cyan] [green]quickstart[/green] [yellow]<name>[/yellow] [dim]\\[directory][/dim]\n\n"
        "[dim]directory optionally overrides the default subdirectory name[/dim]\n\n"
        "[bold]Options:[/bold]\n"
        "  [cyan]--force[/cyan]            Overwrite existing files"
    )
    console.print(Panel(usage_text, title="Usage", border_style="blue"))


@app.command()
def workflow(
    directory: Path = typer.Argument(
        Path("."),
        help="Directory where workflow examples will be created",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force overwrite existing files"),
) -> None:
    """Create workflow pattern examples."""
    target_dir = directory.resolve()
    if not target_dir.exists():
        target_dir.mkdir(parents=True)
        console.print(f"Created directory: {target_dir}")

    created = copy_example_files("workflow", target_dir, force)
    _show_completion_message("workflow", created)


@app.command()
def researcher(
    directory: Path = typer.Argument(
        Path("."),
        help="Directory where researcher examples will be created (in 'researcher' subdirectory)",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force overwrite existing files"),
) -> None:
    """Create researcher pattern examples."""
    target_dir = directory.resolve()
    if not target_dir.exists():
        target_dir.mkdir(parents=True)
        console.print(f"Created directory: {target_dir}")

    created = copy_example_files("researcher", target_dir, force)
    _show_completion_message("researcher", created)


@app.command()
def data_analysis(
    directory: Path = typer.Argument(
        Path("."),
        help="Directory where data analysis examples will be created (creates 'data-analysis' subdirectory with mount-point)",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force overwrite existing files"),
) -> None:
    """Create data analysis examples with sample dataset."""
    target_dir = directory.resolve()
    if not target_dir.exists():
        target_dir.mkdir(parents=True)
        console.print(f"Created directory: {target_dir}")

    created = copy_example_files("data-analysis", target_dir, force)
    _show_completion_message("data-analysis", created)


@app.command()
def state_transfer(
    directory: Path = typer.Argument(
        Path("."),
        help="Directory where state transfer examples will be created (in 'state-transfer' subdirectory)",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force overwrite existing files"),
) -> None:
    """Create state transfer example showing state passing between agents."""
    target_dir = directory.resolve()
    if not target_dir.exists():
        target_dir.mkdir(parents=True)
        console.print(f"Created directory: {target_dir}")

    created = copy_example_files("state-transfer", target_dir, force)
    _show_completion_message("state-transfer", created)


@app.command()
def elicitations(
    directory: Path = typer.Argument(
        Path("."),
        help="Directory where elicitation examples will be created (in 'elicitations' subdirectory)",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force overwrite existing files"),
) -> None:
    """Create interactive form examples using MCP elicitations."""
    target_dir = directory.resolve()
    if not target_dir.exists():
        target_dir.mkdir(parents=True)
        console.print(f"Created directory: {target_dir}")

    created = copy_example_files("elicitations", target_dir, force)
    _show_completion_message("elicitations", created)


def _show_completion_message(example_type: str, created: list[str]) -> None:
    """Show completion message and next steps."""
    if created:
        console.print("\n[green]Setup completed successfully![/green]")
        console.print("\nCreated files:")
        for f in created:
            console.print(f"  - {f}")

        console.print("\n[bold]Next Steps:[/bold]")
        if example_type == "workflow":
            console.print("1. Review chaining.py for the basic workflow example")
            console.print("2. Check other examples:")
            console.print("   - parallel.py: Run agents in parallel")
            console.print("   - router.py: Route requests between agents")
            console.print("   - evaluator.py: Add evaluation capabilities")
            console.print("   - human_input.py: Incorporate human feedback")
            console.print("3. Run an example with: uv run <example>.py")
            console.print(
                "4. Try a different model with --model=<model>, or update the agent config"
            )

        elif example_type == "researcher":
            console.print(
                "1. Set up the Brave MCP Server (get an API key from https://brave.com/search/api/)"
            )
            console.print("2. Try `uv run researcher.py` for the basic version")
            console.print("3. Try `uv run researcher-eval.py` for the eval/optimize version")
        elif example_type == "data-analysis":
            console.print("1. Run uv `analysis.py` to perform data analysis and visualization")
            console.print("2. The dataset is available in the mount-point directory:")
            console.print("   - mount-point/WA_Fn-UseC_-HR-Employee-Attrition.csv")
            console.print(
                "On Windows platforms, please edit the fastagent.config.yaml and adjust the volume mount point."
            )
        elif example_type == "state-transfer":
            console.print(
                "Check [cyan][link=https://fast-agent.ai]fast-agent.ai[/link][/cyan] for quick start walkthroughs"
            )
        elif example_type == "elicitations":
            console.print("1. Go to the `elicitations` subdirectory (cd elicitations)")
            console.print("2. Try the forms demo: uv run forms_demo.py")
            console.print("3. Run the game character creator: uv run game_character.py")
            console.print(
                "Check [cyan][link=https://fast-agent.ai/mcp/elicitations/]https://fast-agent.ai/mcp/elicitations/[/link][/cyan] for more details"
            )
    else:
        console.print("\n[yellow]No files were created.[/yellow]")


@app.command(name="tensorzero", help="Create the TensorZero integration example project.")
def tensorzero(
    directory: Path = typer.Argument(
        Path("."),
        help="Directory where the 'tensorzero' project folder will be created.",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force overwrite if project directory exists"),
):
    """Create the TensorZero project example."""
    console.print("[bold green]Setting up the TensorZero quickstart example...[/bold green]")

    dest_project_dir = directory.resolve() / "tensorzero"

    # --- Find Source Directory ---
    from importlib.resources import files
    try:
        # This path MUST match the "to" path from hatch_build.py
        source_dir = files("mcp_agent").joinpath("resources").joinpath("examples").joinpath("tensorzero")
        if not source_dir.is_dir():
            raise FileNotFoundError  # Fallback to dev mode if resource isn't a dir
    except (ImportError, ModuleNotFoundError, FileNotFoundError):
        console.print("[yellow]Package resources not found. Falling back to development mode.[/yellow]")
        # This path is relative to the project root in a development environment
        source_dir = Path(__file__).parent.parent.parent.parent / "examples" / "tensorzero"

    if not source_dir.exists() or not source_dir.is_dir():
        console.print(f"[red]Error: Source project directory not found at '{source_dir}'[/red]")
        raise typer.Exit(1)

    console.print(f"Source directory: [dim]{source_dir}[/dim]")
    console.print(f"Destination: [dim]{dest_project_dir}[/dim]")

    # --- Copy Project and Show Message ---
    if copy_project_template(source_dir, dest_project_dir, console, force):
        console.print(
            f"\n[bold green]✅ Success![/bold green] Your TensorZero project has been created in: [cyan]{dest_project_dir}[/cyan]"
        )
        console.print("\n[bold yellow]Next Steps:[/bold yellow]")
        console.print("\n1. [bold]Navigate to your new project directory:[/bold]")
        console.print(f"   [cyan]cd {dest_project_dir.relative_to(Path.cwd())}[/cyan]")

        console.print("\n2. [bold]Set up your API keys:[/bold]")
        console.print("   [cyan]cp .env.sample .env[/cyan]")
        console.print(
            "   [dim]Then, open the new '.env' file and add your OpenAI or Anthropic API key.[/dim]"
        )

        console.print("\n3. [bold]Start the required services (TensorZero Gateway & MCP Server):[/bold]")
        console.print("   [cyan]docker compose up --build -d[/cyan]")
        console.print(
            "   [dim](This builds and starts the necessary containers in the background)[/dim]"
        )

        console.print("\n4. [bold]Run the interactive agent:[/bold]")
        console.print("   [cyan]make agent[/cyan]  (or `uv run agent.py`)")
        console.print("\nEnjoy exploring the TensorZero integration with fast-agent! ✨")


@app.command(name="t0", help="Alias for the TensorZero quickstart.", hidden=True)
def t0_alias(
    directory: Path = typer.Argument(Path("."), help="Directory for the 'tensorzero' project folder."),
    force: bool = typer.Option(False, "--force", "-f", help="Force overwrite"),
):
    """Alias for the `tensorzero` command."""
    tensorzero(directory, force)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Quickstart applications for fast-agent."""
    if ctx.invoked_subcommand is None:
        show_overview()
