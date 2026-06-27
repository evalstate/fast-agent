from contextlib import suppress
from pathlib import Path

import typer
from rich.prompt import Confirm

from fast_agent.ui.console import console as shared_console

app = typer.Typer(add_completion=False)
console = shared_console

SCAFFOLD_FILENAMES = (
    "fast-agent.yaml",
    "fast-agent.secrets.yaml",
    "agent.py",
)
PYPROJECT_FILENAME = "pyproject.toml"
GITIGNORE_FILENAME = ".gitignore"
SECRETS_FILENAME = "fast-agent.secrets.yaml"
_SETUP_TEMPLATE_RESOURCE_NAMES: dict[str, str] = {
    SECRETS_FILENAME: "fast-agent.secrets.yaml.example",
    PYPROJECT_FILENAME: "pyproject.toml.tmpl",
}


def _template_resource_name(filename: str) -> str:
    return _SETUP_TEMPLATE_RESOURCE_NAMES.get(filename, filename)


def load_template_text(filename: str) -> str:
    """Load template text from packaged resources only.

    Special-case: when requesting 'fast-agent.secrets.yaml', read the
    'fast-agent.secrets.yaml.example' template from resources, but still
    return its contents so we can write out the real secrets file name
    in the destination project.
    """
    from importlib.resources import files

    res_name = _template_resource_name(filename)
    resource_path = files("fast_agent").joinpath("resources").joinpath("setup").joinpath(res_name)
    if resource_path.is_file():
        return resource_path.read_text()

    raise RuntimeError(
        f"Setup template missing: '{filename}'.\n"
        f"Expected packaged resource at: {resource_path}.\n"
        "This indicates a packaging issue. Please rebuild/reinstall fast-agent."
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


def _ensure_config_dir(config_path: Path) -> None:
    if config_path.exists():
        return

    should_create = Confirm.ask(f"Directory {config_path} does not exist. Create it?", default=True)
    if not should_create:
        raise typer.Abort()
    config_path.mkdir(parents=True)


def _print_scaffold_preview(config_path: Path, *, needs_gitignore: bool) -> None:
    console.print("\n[bold]fast-agent scaffold[/bold]\n")
    console.print("This will create the following files:")
    for filename in (*SCAFFOLD_FILENAMES, PYPROJECT_FILENAME):
        console.print(f"  - {config_path / filename}")
    if needs_gitignore:
        console.print(f"  - {config_path / GITIGNORE_FILENAME}")


def _render_pyproject(template_text: str) -> str:
    # Always use latest fast-agent-mcp (no version pin)
    fast_agent_dep = '"fast-agent-mcp"'

    return template_text.replace("{{python_requires}}", _python_requires()).replace(
        "{{fast_agent_dep}}", fast_agent_dep
    )


def _python_requires() -> str:
    """Return installed package Python requirement, with scaffold fallback."""
    with suppress(Exception):
        from importlib.metadata import metadata

        req = metadata("fast-agent-mcp").get("Requires-Python")
        if req:
            return req
    return ">=3.12"


def _create_scaffold_files(config_path: Path, *, force: bool, needs_gitignore: bool) -> list[str]:
    created = [
        filename
        for filename in SCAFFOLD_FILENAMES
        if create_file(config_path / filename, load_template_text(filename), force)
    ]

    pyproject_text = _render_pyproject(load_template_text(PYPROJECT_FILENAME))
    if create_file(config_path / PYPROJECT_FILENAME, pyproject_text, force):
        created.append(PYPROJECT_FILENAME)

    if needs_gitignore and create_file(
        config_path / GITIGNORE_FILENAME,
        load_template_text(GITIGNORE_FILENAME),
        force,
    ):
        created.append(GITIGNORE_FILENAME)

    return created


def _print_created_file_summary(config_path: Path, created: list[str]) -> None:
    console.print("\n[green]Scaffold completed successfully![/green]")
    console.print(f"Created fast-agent home: {config_path}")
    if "fast-agent.yaml" in created:
        console.print(f"Created config file:     {config_path / 'fast-agent.yaml'}")
    if SECRETS_FILENAME in created:
        console.print(f"Created secrets file:    {config_path / SECRETS_FILENAME}")


def _print_gitignore_note(*, needs_gitignore: bool) -> None:
    if needs_gitignore:
        return

    console.print(
        "[yellow]Note:[/yellow] Found an existing .gitignore in this or a parent directory. "
        f"Ensure it ignores '{SECRETS_FILENAME}' to avoid committing secrets."
    )


def _print_secrets_next_steps(created: list[str]) -> None:
    if SECRETS_FILENAME not in created:
        return

    console.print("\n[yellow]Important:[/yellow] Remember to:")
    console.print(
        "1. Add your API keys to fast-agent.secrets.yaml, or set environment variables. Use [cyan]fast-agent check[/cyan] to verify."
    )
    console.print("2. Keep fast-agent.secrets.yaml secure and never commit it to version control")
    console.print(
        "3. Update fast-agent.yaml to set a default model (currently system default is 'gpt-5-mini?reasoning=low')"
    )


def _print_scaffold_result(
    config_path: Path,
    created: list[str],
    *,
    needs_gitignore: bool,
) -> None:
    if not created:
        console.print("\n[yellow]No files were created or modified.[/yellow]")
        return

    _print_created_file_summary(config_path, created)
    _print_gitignore_note(needs_gitignore=needs_gitignore)
    _print_secrets_next_steps(created)
    console.print("\nTo get started, run:")
    console.print("  uv run agent.py")


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
    _ensure_config_dir(config_path)

    # Check for existing .gitignore
    needs_gitignore = not find_gitignore(config_path)

    _print_scaffold_preview(config_path, needs_gitignore=needs_gitignore)

    if not Confirm.ask("\nContinue?", default=True):
        raise typer.Abort()

    created = _create_scaffold_files(
        config_path,
        force=force,
        needs_gitignore=needs_gitignore,
    )
    _print_scaffold_result(config_path, created, needs_gitignore=needs_gitignore)
