"""Shared constants for CLI routing and commands."""

# Options that should automatically route to the 'go' command
GO_SPECIFIC_OPTIONS = {
    "--npx",
    "--uvx",
    "--stdio",
    "--url",
    "--model",
    "--models",
    "--instruction",
    "-i",
    "--message",
    "-m",
    "--prompt-file",
    "-p",
    "--servers",
    "--auth",
    "--name",
    "--config-path",
    "-c",
    "--shell",
    "-x",
    "--skills",
    "--skills-dir",
}

# Known subcommands that should not trigger auto-routing
KNOWN_SUBCOMMANDS = {
    "go",
    "serve",
    "setup",
    "check",
    "auth",
    "bootstrap",
    "quickstart",
    "--help",
    "-h",
    "--version",
}
