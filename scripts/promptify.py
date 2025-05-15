# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "rich",
#     "typer",
# ]
# ///
"""
Convert the project directory structure and file contents into a single markdown file.
Really helpful for using as a prompt for LLM code generation tasks.
"""

import fnmatch
from pathlib import Path
from typing import List

import typer
from rich.console import Console
from rich.tree import Tree


def parse_gitignore(path: Path) -> List[str]:
    """Parse .gitignore file and return list of patterns."""
    gitigore_path = path / ".gitignore"
    if not gitigore_path.exists():
        return []

    with open(file=gitigore_path, mode="r", encoding="utf-8") as f:
        patterns = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return patterns


def normalize_pattern(pattern: str) -> str:
    """
    Normalize a pattern by removing unnecessary whitespace and escaping special characters.
    """
    # Strip whitespace
    pattern = pattern.strip()
    return pattern


def pattern_match(path: str, pattern: str) -> bool:
    """
    Improved pattern matching that better handles **/ patterns and different path separators.
    """
    # Normalize the pattern first
    pattern = normalize_pattern(pattern)
    path = path.replace("\\", "/")  # Normalize path separators

    # Handle **/ prefix more flexibly
    if pattern.startswith("**/"):
        base_pattern = pattern[3:]  # Pattern without **/ prefix
        # Try matching both with and without the **/ prefix
        return fnmatch.fnmatch(path, base_pattern) or fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, f"**/{base_pattern}")

    # Handle *registry.py style patterns
    elif pattern.startswith("*") and not pattern.startswith("**/"):
        return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, f"**/{pattern}")

    return fnmatch.fnmatch(path, pattern)


def should_include(path: Path, include_patterns: List[str]) -> bool:
    """Check if path should be included based on patterns."""
    if not include_patterns:
        return True

    str_path = str(path).replace("\\", "/")

    # For directories, we want to include them if they might contain matching files
    if path.is_dir():
        # If any pattern starts with the directory path, include it
        dir_path = str_path + "/"
        for pattern in include_patterns:
            pattern = normalize_pattern(pattern)
            if pattern.startswith("**/"):
                # Always include directories when we have **/ patterns
                return True
            # Check if this directory might contain matching files
            if fnmatch.fnmatch(dir_path + "anyfile", pattern):
                return True
        return False

    # For files, check against all patterns
    return any(pattern_match(str_path, p) for p in include_patterns)


def should_ignore(path: Path, ignore_patterns: List[str], gitignore_patterns: List[str]) -> bool:
    """Check if path should be ignored based on patterns."""
    str_path = str(path).replace("\\", "/")

    # Check custom ignore patterns
    for pattern in ignore_patterns:
        if pattern_match(str_path, pattern):
            return True

    # Check gitignore patterns
    for pattern in gitignore_patterns:
        if pattern_match(str_path, pattern):
            return True

    return False


def create_tree_structure(
    path: Path,
    include_patterns: List[str],
    ignore_patterns: List[str],
    gitignore_patterns: List[str],
) -> Tree:
    """Create a rich Tree representation of the directory structure."""
    tree = Tree(f"📁 {path.name}")

    def add_to_tree(current_path: Path, tree: Tree):
        items = sorted(current_path.iterdir())
        for item in items:
            # Skip if should be ignored
            if should_ignore(item, ignore_patterns, gitignore_patterns):
                continue

            # Check if item matches include patterns (if any)
            if not should_include(item, include_patterns):
                continue

            if item.is_file():
                tree.add(f"📄 {item.name}")
            elif item.is_dir():
                branch = tree.add(f"📁 {item.name}")
                add_to_tree(item, branch)

    add_to_tree(path, tree)
    return tree


def package_project(
    path: Path,
    output_file: Path,
    include_patterns: List[str],
    ignore_patterns: List[str],
    gitignore_patterns: List[str],
) -> None:
    """Package project files into a single markdown file."""
    # Normalize all patterns first
    include_patterns = [normalize_pattern(p) for p in include_patterns]
    ignore_patterns = [normalize_pattern(p) for p in ignore_patterns]
    gitignore_patterns = [normalize_pattern(p) for p in gitignore_patterns]

    with open(output_file, "w", encoding="utf-8") as f:
        # Write header
        f.write(f"# Project: {path.name}\n\n")

        # Write directory structure
        f.write("## Directory Structure\n\n")
        f.write("```\n")
        console = Console(file=None)
        with console.capture() as capture:
            console.print(create_tree_structure(path, include_patterns, ignore_patterns, gitignore_patterns))
        f.write(capture.get())
        f.write("```\n\n")

        # Write file contents
        f.write("## File Contents\n\n")

        def write_files(current_path: Path):
            for item in sorted(current_path.iterdir()):
                if should_ignore(item, ignore_patterns, gitignore_patterns):
                    continue

                if include_patterns and not should_include(item, include_patterns):
                    continue

                if item.is_file():
                    try:
                        with open(item, "r", encoding="utf-8") as source_file:
                            content = source_file.read()
                            f.write(f"### {item.relative_to(path)}\n\n")
                            f.write("```")
                            # Add file extension for syntax highlighting if available
                            if item.suffix:
                                f.write(item.suffix[1:])  # Remove the dot from extension
                            f.write("\n")
                            f.write(content)
                            f.write("\n```\n\n")
                    except UnicodeDecodeError:
                        f.write(f"### {item.relative_to(path)}\n\n")
                        f.write("```\nBinary file not included\n```\n\n")
                elif item.is_dir():
                    write_files(item)

        write_files(path)


def main(
    path: str = typer.Argument(".", help="Path to the project directory"),
    output: str = typer.Option("project_contents.md", "--output", "-o", help="Output file path"),
    include: List[str] | None = typer.Option(None, "--include", "-i", help="Patterns to include (e.g. '*.py')"),
    ignore: List[str] | None = typer.Option(None, "--ignore", "-x", help="Patterns to ignore"),
    skip_gitignore: bool = typer.Option(False, "--skip-gitignore", help="Skip reading .gitignore patterns"),
) -> None:
    """
    Package project files into a single markdown file with directory structure.
    """
    project_path = Path(path).resolve()
    output_path = Path(output).resolve()

    if not project_path.exists():
        typer.echo(f"Error: Project path '{path}' does not exist")
        raise typer.Exit(1)

    # Parse .gitignore if needed
    gitignore_patterns = [] if skip_gitignore else parse_gitignore(project_path)

    # Convert None to empty lists
    include_patterns = include or []
    ignore_patterns = ignore or []

    # Add some default ignore patterns
    # Default ignore patterns for Python development
    default_ignores = [
        "**/__pycache__/**",
        "**/.git/**",
        "**/.idea/**",
        "**/.vscode/**",
        "**/.ruff_cache/**",
        "**/.venv/**",
        "**/venv/**",
        "**/env/**",
        "**/uv.lock",
        "**/.pytest_cache/**",
        "**/*.pyc",
        "**/.coverage",
        "**/htmlcov/**",
    ]
    ignore_patterns.extend(default_ignores)

    try:
        package_project(
            project_path,
            output_path,
            include_patterns,
            ignore_patterns,
            gitignore_patterns,
        )
        typer.echo(f"Successfully packaged project to {output_path}")
    except Exception as e:
        typer.echo(f"Error packaging project: {str(e)}")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
