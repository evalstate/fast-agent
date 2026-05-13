"""Optional privacy-filter dependency checks."""

from __future__ import annotations

from importlib.util import find_spec

PRIVACY_EXTRA_REQUIREMENTS = {
    "onnxruntime": "onnxruntime",
    "tokenizers": "tokenizers",
    "numpy": "numpy",
}

PRIVACY_EXTRA_INSTALL_MESSAGE = (
    'Install fast-agent with the privacy extra, "fast-agent-mcp[privacy]", '
    "in the environment where fast-agent runs."
)


def missing_privacy_dependencies() -> list[str]:
    """Return package names missing for privacy-filter inference."""

    return [
        package
        for module, package in PRIVACY_EXTRA_REQUIREMENTS.items()
        if find_spec(module) is None
    ]


def format_missing_privacy_dependencies(packages: list[str]) -> str:
    """Render an actionable missing-dependency error."""

    package_lines = "\n".join(f"  - {package}" for package in packages)
    return (
        "Privacy filtering requires optional dependencies that are not installed:\n"
        f"{package_lines}\n\n"
        f"{PRIVACY_EXTRA_INSTALL_MESSAGE}"
    )
