#!/usr/bin/env python3
"""Sync version from main pyproject.toml to fast-agent-acp wrapper.

This script ensures that the fast-agent-acp wrapper package always:
1. Has the same version as fast-agent-mcp
2. Pins its dependency to the exact version of fast-agent-mcp

This is automatically run during the GitHub Actions publish workflow,
but can also be run manually before local builds:

    python3 scripts/sync-wrapper-version.py
"""

import tomllib
from pathlib import Path


def sync_version():
    """Sync version and dependency from main package to wrapper."""
    root = Path(__file__).parent.parent
    main_pyproject = root / "pyproject.toml"
    wrapper_pyproject = root / "publish" / "fast-agent-acp" / "pyproject.toml"

    # Read main version
    with open(main_pyproject, "rb") as f:
        main_config = tomllib.load(f)

    version = main_config["project"]["version"]

    # Read wrapper config
    with open(wrapper_pyproject, "rb") as f:
        wrapper_config = tomllib.load(f)

    # Update wrapper pyproject.toml
    wrapper_text = wrapper_pyproject.read_text()

    # Update version
    old_version = wrapper_config["project"]["version"]
    wrapper_text = wrapper_text.replace(
        f'version = "{old_version}"',
        f'version = "{version}"'
    )

    # Update dependency to pin to exact version
    wrapper_text = wrapper_text.replace(
        'dependencies = [\n    "fast-agent-mcp",\n]',
        f'dependencies = [\n    "fast-agent-mcp=={version}",\n]'
    )

    # Also handle case where it's already pinned
    import re
    wrapper_text = re.sub(
        r'"fast-agent-mcp[^"]*"',
        f'"fast-agent-mcp=={version}"',
        wrapper_text
    )

    wrapper_pyproject.write_text(wrapper_text)
    print(f"Updated wrapper version to {version}")
    print(f"Pinned fast-agent-mcp dependency to =={version}")


if __name__ == "__main__":
    sync_version()
