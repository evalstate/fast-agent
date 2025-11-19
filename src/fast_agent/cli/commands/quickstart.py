"""Bootstrap command to create example applications."""

import shutil
from contextlib import ExitStack
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fast_agent.ui.console import console as shared_console

app = typer.Typer(
    help="Create fast-agent quickstarts",
    no_args_is_help=False,  # Allow showing our custom help instead
)
console = shared_console


BASE_EXAMPLES_DIR = files("fast_agent").joinpath("resources").joinpath("examples")


@dataclass
class ExampleConfig:
    description: str
    files: list[str]
    create_subdir: bool
    path_in_examples: list[str]
    mount_point_files: list[str] | None = None


_EXAMPLE_CONFIGS = {
    "workflow": ExampleConfig(
        description=(
            "Example workflows, demonstrating each of the patterns in Anthropic's\n"
            "'Building Effective Agents' paper. Some agents use the 'fetch'\n"
            "and filesystem MCP Servers."
        ),
        files=[
            "chaining.py",
            "evaluator.py",
            "human_input.py",
            "orchestrator.py",
            "parallel.py",
            "router.py",
            "short_story.txt",
            "fastagent.config.yaml",
        ],
        create_subdir=True,
        path_in_examples=["workflows"],
    ),
    "researcher": ExampleConfig(
        description=(
            "Research agent example with additional evaluation/optimization\n"
            "example. Uses Brave Search and Docker MCP Servers.\n"
            "Creates examples in a 'researcher' subdirectory."
        ),
        files=["researcher.py", "researcher-eval.py", "fastagent.config.yaml"],
        create_subdir=True,
        path_in_examples=["researcher"],
    ),
    "data-analysis": ExampleConfig(
        description=(
            "Data analysis agent examples that demonstrate working with\n"
            "datasets, performing statistical analysis, and generating visualizations.\n"
            "Creates examples in a 'data-analysis' subdirectory with mount-point for data.\n"
            "Uses MCP 'roots' feature for mapping"
        ),
        files=["analysis.py", "fastagent.config.yaml"],
        mount_point_files=["WA_Fn-UseC_-HR-Employee-Attrition.csv"],
        create_subdir=True,
        path_in_examples=["data-analysis"],
    ),
    "state-transfer": ExampleConfig(
        description=(
            "Example demonstrating state transfer between multiple agents.\n"
            "Shows how state can be passed between agent runs to maintain context.\n"
            "Creates examples in a 'state-transfer' subdirectory."
        ),
        files=[
            "agent_one.py",
            "agent_two.py",
            "fastagent.config.yaml",
            "fastagent.secrets.yaml.example",
        ],
        create_subdir=True,
        path_in_examples=["mcp", "state-transfer"],
    ),
    "elicitations": ExampleConfig(
        description=(
            "Interactive form examples using MCP elicitations feature.\n"
            "Demonstrates collecting structured data with forms, AI-guided workflows,\n"
            "and custom handlers. Creates examples in an 'elicitations' subdirectory."
        ),
        files=[
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
        create_subdir=True,
        path_in_examples=["mcp", "elicitations"],
    ),
    "tensorzero": ExampleConfig(
        description=(
            "A complete example showcasing the TensorZero integration.\n"
            "Includes the T0 Gateway, an MCP server, an interactive agent, and \n"
            "multi-modal functionality."
        ),
        files=[
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
            "tensorzero_config/",
        ],
        create_subdir=True,
        path_in_examples=["elicitations"],
    ),
}


def _development_mode_fallback(example_info: ExampleConfig) -> Path:
    """Fallback function for development mode."""
    package_dir = Path(__file__).parent.parent.parent.parent.parent
    for dir in example_info.path_in_examples:
        package_dir = package_dir / dir
    console.print(f"[blue]Using development directory: {package_dir}[/blue]")
    return package_dir


def copy_example_files(example_type: str, target_dir: Path, force: bool = False) -> list[str]:
    """Copy example files from resources to target directory."""
    # Determine if we should create a subdirectory for this example type
    example_info = _EXAMPLE_CONFIGS.get(example_type, None)
    if example_info is None:
        console.print(f"Example type '{example_type}' not found.")
        return []

    if example_info.create_subdir:
        target_dir = target_dir / example_type
        if not target_dir.exists():
            target_dir.mkdir(parents=True)
            console.print(f"Created subdirectory: {target_dir}")

    # Determine source directory - try package resources first, then fallback
    use_as_file = False
    # Try to use examples from the installed package first, or fall back to the top-level directory
    try:
        # First try to find examples in the package resources
        source_dir_traversable = BASE_EXAMPLES_DIR
        for dir in example_info.path_in_examples:
            source_dir_traversable = source_dir_traversable.joinpath(dir)

        # Check if we found a valid directory
        if not source_dir_traversable.is_dir():
            console.print(
                f"[yellow]Resource directory not found: {source_dir_traversable}. "
                "Falling back to development mode.[/yellow]"
            )
            # Fall back to the top-level directory for development mode
            source_dir: Path = _development_mode_fallback(example_info)
        else:
            # We have a valid Traversable, will need to use as_file
            source_dir = source_dir_traversable  # type: ignore
            use_as_file = True
    except (ImportError, ModuleNotFoundError, ValueError) as e:
        console.print(
            f"[yellow]Error accessing resources: {e}. Falling back to development mode.[/yellow]"
        )
        source_dir = _development_mode_fallback(example_info)

    # Use as_file context manager if source_dir is a Traversable, otherwise use directly
    with ExitStack() as stack:
        if use_as_file:
            source_path = stack.enter_context(as_file(source_dir))  # type: ignore
        else:
            source_path = source_dir

        if not source_path.exists():
            console.print(f"[red]Error: Source directory not found: {source_path}[/red]")
            return []

        return _copy_files_from_source(
            example_type, example_info, source_path, target_dir, force
        )


def _copy_files_from_source(
    example_type: str, example_info: ExampleConfig, source_dir: Path, target_dir: Path, force: bool
) -> list[str]:
    """Helper function to copy files from a source directory."""
    created = []
    for filename in example_info.files:
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
            console.print(f"[green]Created[/green] {rel_path}")

        except Exception as e:
            console.print(f"[red]Error copying {filename}: {str(e)}[/red]")

    # Copy mount-point files if any
    mount_point_files = example_info.mount_point_files or []
    if mount_point_files:
        mount_point_dir = target_dir / "mount-point"

        # Create mount-point directory if needed
        if not mount_point_dir.exists():
            mount_point_dir.mkdir(parents=True)
            console.print(f"Created mount-point directory: {mount_point_dir}")

        for filename in mount_point_files:
            source = source_dir / "mount-point" / filename
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
            console.print(
                f"[yellow]--force specified. Removing existing directory: {dest_dir}[/yellow]"
            )
            shutil.rmtree(dest_dir)
        else:
            console.print(
                f"[bold yellow]Directory '{dest_dir.name}' already exists.[/bold yellow] Use --force to overwrite."
            )
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

    for name, info in _EXAMPLE_CONFIGS.items():
        # Just show file count instead of listing all files
        file_count = len(info.files)
        files_summary = f"{file_count} files"
        mount_files = info.mount_point_files
        if mount_files:
            files_summary += f"\n+ {len(mount_files)} data files"
        table.add_row(f"[green]{name}[/green]", info.description, files_summary)

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
    force: bool = typer.Option(
        False, "--force", "-f", help="Force overwrite if project directory exists"
    ),
):
    """Create the TensorZero project example."""
    console.print("[bold green]Setting up the TensorZero quickstart example...[/bold green]")

    dest_project_dir = directory.resolve() / "tensorzero"

    # --- Find Source Directory ---
    use_as_file = False
    try:
        # This path MUST match the "to" path from hatch_build.py
        source_dir_traversable = (
            files("fast_agent").joinpath("resources").joinpath("examples").joinpath("tensorzero")
        )
        if not source_dir_traversable.is_dir():
            raise FileNotFoundError  # Fallback to dev mode if resource isn't a dir
        source_dir = source_dir_traversable  # type: ignore
        use_as_file = True
    except (ImportError, ModuleNotFoundError, FileNotFoundError):
        console.print(
            "[yellow]Package resources not found. Falling back to development mode.[/yellow]"
        )
        # This path is relative to the project root in a development environment
        source_dir = Path(__file__).parent.parent.parent.parent / "examples" / "tensorzero"

    # Use as_file context manager if needed
    with ExitStack() as stack:
        if use_as_file:
            source_path = stack.enter_context(as_file(source_dir))  # type: ignore
        else:
            source_path = source_dir  # type: ignore[assignment]

        if not source_path.exists() or not source_path.is_dir():
            console.print(f"[red]Error: Source project directory not found at '{source_path}'[/red]")
            raise typer.Exit(1)

        console.print(f"Source directory: [dim]{source_path}[/dim]")
        console.print(f"Destination: [dim]{dest_project_dir}[/dim]")

        # --- Copy Project and Show Message ---
        if copy_project_template(source_path, dest_project_dir, console, force):
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

            console.print(
                "\n3. [bold]Start the required services (TensorZero Gateway & MCP Server):[/bold]"
            )
            console.print("   [cyan]docker compose up --build -d[/cyan]")
            console.print(
                "   [dim](This builds and starts the necessary containers in the background)[/dim]"
            )

            console.print("\n4. [bold]Run the interactive agent:[/bold]")
            console.print("   [cyan]make agent[/cyan]  (or `uv run agent.py`)")
            console.print("\nEnjoy exploring the TensorZero integration with fast-agent! ✨")


@app.command(name="t0", help="Alias for the TensorZero quickstart.", hidden=True)
def t0_alias(
    directory: Path = typer.Argument(
        Path("."), help="Directory for the 'tensorzero' project folder."
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force overwrite"),
):
    """Alias for the `tensorzero` command."""
    tensorzero(directory, force)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Quickstart applications for fast-agent."""
    if ctx.invoked_subcommand is None:
        show_overview()
