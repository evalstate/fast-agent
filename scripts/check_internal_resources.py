from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich import print

try:
    from fast_agent.core.exceptions import AgentConfigError
    from fast_agent.core.internal_resources import list_internal_resources, read_internal_resource
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from fast_agent.core.exceptions import AgentConfigError
    from fast_agent.core.internal_resources import list_internal_resources, read_internal_resource


def validate_internal_resources() -> list[str]:
    """Validate manifest entries and ensure each resource can be read."""
    resources = list_internal_resources()
    validated_uris: list[str] = []

    for resource in resources:
        content = read_internal_resource(resource.uri)
        if not content.strip():
            raise AgentConfigError(
                "Internal resource is empty",
                f"URI: {resource.uri}",
            )
        validated_uris.append(resource.uri)

    return validated_uris


def main(verbose: bool = False) -> None:
    try:
        validated = validate_internal_resources()
    except AgentConfigError as exc:
        print(f"[red]Internal resource validation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:  # noqa: BLE001
        print(f"[red]Unexpected validation error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    count = len(validated)
    print(f"[green]Internal resources valid:[/green] {count}")
    if verbose:
        for uri in validated:
            print(f"  - {uri}")

    raise typer.Exit(code=0)


if __name__ == "__main__":
    typer.run(main)
