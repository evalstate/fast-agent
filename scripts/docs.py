#!/usr/bin/env python3
"""
Documentation generation and serving utilities.

Usage:
    uv run scripts/docs.py install    # Install docs dependencies
    uv run scripts/docs.py generate   # Generate reference docs from source
    uv run scripts/docs.py serve      # Run mkdocs dev server
    uv run scripts/docs.py build      # Build static site
    uv run scripts/docs.py all        # Generate + serve
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"


def install() -> int:
    """Install documentation dependencies using uv."""
    print("Installing docs dependencies...")
    result = subprocess.run(
        ["uv", "pip", "install", "-r", str(DOCS_DIR / "requirements.txt")],
        cwd=ROOT,
    )
    if result.returncode == 0:
        print("Docs dependencies installed successfully.")
    return result.returncode


def generate() -> int:
    """Generate reference documentation from fast-agent source."""
    print("Generating reference docs...")
    result = subprocess.run(
        [sys.executable, str(DOCS_DIR / "generate_reference_docs.py")],
        cwd=ROOT,
    )
    if result.returncode == 0:
        print(f"Generated docs in {DOCS_DIR / 'docs' / '_generated'}")
    return result.returncode


def serve() -> int:
    """Run mkdocs development server."""
    print(f"Starting mkdocs server from {DOCS_DIR}...")
    print("Site will be available at http://127.0.0.1:8000")
    result = subprocess.run(
        ["mkdocs", "serve"],
        cwd=DOCS_DIR,
    )
    return result.returncode


def build() -> int:
    """Build static documentation site."""
    print(f"Building static site from {DOCS_DIR}...")
    result = subprocess.run(
        ["mkdocs", "build"],
        cwd=DOCS_DIR,
    )
    if result.returncode == 0:
        print(f"Built site in {DOCS_DIR / 'site'}")
    return result.returncode


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    command = sys.argv[1]

    if command == "install":
        return install()
    elif command == "generate":
        return generate()
    elif command == "serve":
        return serve()
    elif command == "build":
        return generate() or build()
    elif command == "all":
        return generate() or serve()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
