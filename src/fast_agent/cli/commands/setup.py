from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm

app = typer.Typer()
console = Console()


def load_template_text(filename: str) -> str:
    """Load template text from packaged resources with a development fallback.

    Special-case: when requesting 'fastagent.secrets.yaml', read the
    'fastagent.secrets.yaml.example' template from resources, but still
    return its contents so we can write out the real secrets file name
    in the destination project.
    """
    try:
        # Prefer reading from installed package resources
        from importlib.resources import files

        # Map secrets filename to its '.example' template in resources
        res_name = (
            "fastagent.secrets.yaml.example" if filename == "fastagent.secrets.yaml" else filename
        )
        resource_path = files("fast_agent").joinpath("resources").joinpath("setup").joinpath(res_name)
        if resource_path.is_file():
            return resource_path.read_text()
        # If the path exists but isn't a file, fall back to dev
        raise FileNotFoundError
    except (ImportError, ModuleNotFoundError, FileNotFoundError):
        # Development environment fallback to repo path (project root/examples/setup)
        try:
            repo_root = Path(__file__).resolve().parents[4]
        except IndexError:
            repo_root = Path(__file__).resolve().parent
        # Apply the same secrets mapping for the dev path
        dev_name = (
            "fastagent.secrets.yaml.example" if filename == "fastagent.secrets.yaml" else filename
        )
        dev_path = repo_root / "examples" / "setup" / dev_name
        if dev_path.exists():
            return dev_path.read_text()
        raise RuntimeError(
            "Setup template missing: '"
            + filename
            + "'.\n"
            + "Expected at: "
            + str(resource_path)
            + " (package) or "
            + str(dev_path)
            + " (dev).\n"
            + "This indicates a packaging issue. Please rebuild/reinstall fast-agent."
        )


# (No embedded template defaults; templates are the single source of truth.)


def find_gitignore(path: Path) -> bool:
    """Check if a .gitignore file exists in this directory or any parent."""
    current = path
    while current != current.parent:  # Stop at root directory
        if (current / ".gitignore").exists():
            return True
        current = current.parent
    return False


def create_file(path: Path, content: str, force: bool = False) -> bool:
    """Create a file with given content if it doesn't exist or force is True."""
    if path.exists() and not force:
        should_overwrite = Confirm.ask(
            f"[yellow]Warning:[/yellow] {path} already exists. Overwrite?",
            default=False,
        )
        if not should_overwrite:
            console.print(f"Skipping {path}")
            return False

    path.write_text(content.strip() + "\n")
    console.print(f"[green]Created[/green] {path}")
    return True


@app.callback(invoke_without_command=True)
def init(
    config_dir: str = typer.Option(
        ".",
        "--config-dir",
        "-c",
        help="Directory where configuration files will be created",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Force overwrite existing files"),
) -> None:
    """Initialize a new FastAgent project with configuration files and example agent."""

    config_path = Path(config_dir).resolve()
    if not config_path.exists():
        should_create = Confirm.ask(
            f"Directory {config_path} does not exist. Create it?", default=True
        )
        if should_create:
            config_path.mkdir(parents=True)
        else:
            raise typer.Abort()

    # Check for existing .gitignore
    needs_gitignore = not find_gitignore(config_path)

    console.print("\n[bold]fast-agent setup[/bold]\n")
    console.print("This will create the following files:")
    console.print(f"  - {config_path}/fastagent.config.yaml")
    console.print(f"  - {config_path}/fastagent.secrets.yaml")
    console.print(f"  - {config_path}/agent.py")
    if needs_gitignore:
        console.print(f"  - {config_path}/.gitignore")

    if not Confirm.ask("\nContinue?", default=True):
        raise typer.Abort()

    # Create configuration files
    created = []
    if create_file(
        config_path / "fastagent.config.yaml", load_template_text("fastagent.config.yaml"), force
    ):
        created.append("fastagent.yaml")

    if create_file(
        config_path / "fastagent.secrets.yaml", load_template_text("fastagent.secrets.yaml"), force
    ):
        created.append("fastagent.secrets.yaml")

    if create_file(config_path / "agent.py", load_template_text("agent.py"), force):
        created.append("agent.py")

    # Only create .gitignore if none exists in parent directories
    if needs_gitignore and create_file(
        config_path / ".gitignore", load_template_text(".gitignore"), force
    ):
        created.append(".gitignore")

    if created:
        console.print("\n[green]Setup completed successfully![/green]")
        if not needs_gitignore:
            console.print(
                "[yellow]Note:[/yellow] Found an existing .gitignore in this or a parent directory. "
                "Ensure it ignores 'fastagent.secrets.yaml' to avoid committing secrets."
            )
        if "fastagent.secrets.yaml" in created:
            console.print("\n[yellow]Important:[/yellow] Remember to:")
            console.print(
                "1. Add your API keys to fastagent.secrets.yaml, or set environment variables. Use [cyan]fast-agent check[/cyan] to verify."
            )
            console.print(
                "2. Keep fastagent.secrets.yaml secure and never commit it to version control"
            )
            console.print(
                "3. Update fastagent.config.yaml to set a default model (currently system default is 'haiku')"
            )
        console.print("\nTo get started, run:")
        console.print("  uv run agent.py")
    else:
        console.print("\n[yellow]No files were created or modified.[/yellow]")
