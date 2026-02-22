"""Shared constants for CLI routing and commands."""

RESUME_LATEST_SENTINEL = "__latest__"


def normalize_resume_flag_args(args: list[str], *, start_index: int = 0) -> None:
    index = start_index
    while index < len(args):
        arg = args[index]
        if arg == "--resume":
            next_arg = args[index + 1] if index + 1 < len(args) else None
            if next_arg is None or next_arg.startswith("-"):
                args.insert(index + 1, RESUME_LATEST_SENTINEL)
                index += 1
        index += 1


# Options that should automatically route to the 'go' command
GO_SPECIFIC_OPTIONS = {
    "--npx",
    "--uvx",
    "--stdio",
    "--url",
    "--model",
    "--models",
    "--agent",
    "--instruction",
    "-i",
    "--message",
    "-m",
    "--prompt-file",
    "-p",
    "--results",
    "--servers",
    "--auth",
    "--client-metadata-url",
    "--name",
    "--config-path",
    "-c",
    "--shell",
    "-x",
    "--skills",
    "--skills-dir",
    "--agent-cards",
    "--card",
    "--env",
    "--noenv",
    "--no-env",
    "--watch",
    "--reload",
    "--resume",
    "--smart",
}

# Known subcommands that should not trigger auto-routing
KNOWN_SUBCOMMANDS = {
    "go",
    "serve",
    "acp",
    "setup",
    "check",
    "auth",
    "bootstrap",
    "quickstart",
    "--help",
    "-h",
    "--version",
}
