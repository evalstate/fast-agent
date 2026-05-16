#!/usr/bin/env python3
"""
Documentation generation and serving utilities.

Usage:
    uv run scripts/docs.py install    # Install/sync dev dependencies
    uv run scripts/docs.py generate   # Generate reference docs from source
    uv run scripts/docs.py serve      # Run Zensical dev server
    uv run scripts/docs.py build      # Build static site
    uv run scripts/docs.py screenshot # Capture local and live docs screenshots
    uv run scripts/docs.py assess     # Run deterministic visual screenshot checks
    uv run scripts/docs.py all        # Generate + serve
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
SCREENSHOT_DIR = DOCS_DIR / "screenshots"


def _run_docs_tool(*args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["uv", "run", *args], cwd=DOCS_DIR)


def install() -> int:
    """Install/sync documentation dependencies using uv."""
    print("Syncing development dependencies...")
    result = subprocess.run(["uv", "sync", "--group", "dev"], cwd=ROOT)
    if result.returncode == 0:
        print("Docs dependencies synced successfully.")
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
    """Run Zensical development server."""
    print(f"Starting Zensical server from {DOCS_DIR}...")
    print("Site will be available at http://127.0.0.1:8000")
    result = _run_docs_tool("zensical", "serve")
    return result.returncode


def build() -> int:
    """Build static documentation site."""
    print(f"Building static site from {DOCS_DIR}...")
    result = _run_docs_tool("zensical", "build", "--strict")
    if result.returncode == 0:
        print(f"Built site in {DOCS_DIR / 'site'}")
    return result.returncode


def screenshot() -> int:
    """Capture comparison screenshots using google-chrome."""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    targets = [
        ("live-home.png", "https://fast-agent.ai", "1440,1200"),
        ("local-home.png", f"file://{(DOCS_DIR / 'site' / 'index.html').resolve()}", "1440,1200"),
        (
            "local-home-mobile.png",
            f"file://{(DOCS_DIR / 'site' / 'index.html').resolve()}",
            "390,900",
        ),
        (
            "local-models.png",
            f"file://{(DOCS_DIR / 'site' / 'models' / 'llm_providers' / 'index.html').resolve()}",
            "1440,1200",
        ),
    ]
    for filename, url, window_size in targets:
        output = SCREENSHOT_DIR / filename
        print(f"Capturing {url} -> {output}")
        result = subprocess.run(
            [
                "google-chrome",
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                f"--window-size={window_size}",
                f"--screenshot={output}",
                url,
            ],
            cwd=ROOT,
        )
        if result.returncode != 0:
            return result.returncode
    return 0


def assess() -> int:
    """Run deterministic visual checks for captured documentation screenshots."""
    result = subprocess.run(
        [
            "uv",
            "run",
            "scripts/docs_visual_assess.py",
            "--screenshots-dir",
            str(SCREENSHOT_DIR),
        ],
        cwd=ROOT,
    )
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
    elif command == "screenshot":
        return build() or screenshot()
    elif command == "assess":
        return assess()
    elif command == "all":
        return generate() or serve()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
