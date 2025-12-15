#!/usr/bin/env python3
"""Sync version from main pyproject.toml to wrapper packages.

This script ensures that the wrapper packages always:
1. Have the same version as fast-agent-mcp
2. Pin their fast-agent-mcp dependency to the exact version of fast-agent-mcp

This is automatically run during the GitHub Actions publish workflow,
but can also be run manually before local builds:

    python3 scripts/sync-wrapper-version.py
"""

import tomllib
from pathlib import Path
from typing import Iterable


def _sync_wrapper(wrapper_pyproject: Path, main_version: str) -> None:
    """Sync a wrapper's version and dependency pin."""
    with open(wrapper_pyproject, "rb") as f:
        wrapper_config = tomllib.load(f)

    wrapper_text = wrapper_pyproject.read_text()

    # Update version
    old_version = wrapper_config["project"]["version"]
    wrapper_text = wrapper_text.replace(
        f'version = "{old_version}"',
        f'version = "{main_version}"',
    )

    # Update dependency to pin to exact version
    wrapper_text = wrapper_text.replace(
        'dependencies = [\n    "fast-agent-mcp",\n]',
        f'dependencies = [\n    "fast-agent-mcp=={main_version}",\n]',
    )

    # Also handle case where it's already pinned
    import re

    wrapper_text = re.sub(
        r'"fast-agent-mcp[^"]*"',
        f'"fast-agent-mcp=={main_version}"',
        wrapper_text,
    )

    wrapper_pyproject.write_text(wrapper_text)
    print(f"Updated {wrapper_pyproject} to version {main_version} (pin fast-agent-mcp=={main_version})")


def sync_version():
    """Sync version and dependency from main package to wrappers."""
    root = Path(__file__).parent.parent
    main_pyproject = root / "pyproject.toml"
    wrapper_projects: Iterable[Path] = [
        root / "publish" / "fast-agent-acp" / "pyproject.toml",
        root / "publish" / "hf-inference-acp" / "pyproject.toml",
    ]

    # Read main version
    with open(main_pyproject, "rb") as f:
        main_config = tomllib.load(f)

    version = main_config["project"]["version"]

    for wrapper_pyproject in wrapper_projects:
        if wrapper_pyproject.exists():
            _sync_wrapper(wrapper_pyproject, version)
        else:
            print(f"Warning: wrapper pyproject not found: {wrapper_pyproject}")


if __name__ == "__main__":
    sync_version()
