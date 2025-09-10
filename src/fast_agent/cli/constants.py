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
}

# Known subcommands that should not trigger auto-routing
KNOWN_SUBCOMMANDS = {"go", "setup", "check", "bootstrap", "quickstart", "--help", "-h", "--version"}
