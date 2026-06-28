"""Shared constants for CLI routing and commands."""

RESUME_LATEST_SENTINEL = "__latest__"


def normalize_resume_flag_args(args: list[str], *, start_index: int = 0) -> None:
    index = start_index
    while index < len(args):
        arg = args[index]
        if arg == "--resume=":
            args[index] = "--resume"
            args.insert(index + 1, RESUME_LATEST_SENTINEL)
            index += 1
        elif arg == "--resume":
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
    "--pack",
    "--card-pack",
    "--pack-registry",
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
    "--attach",
    "-a",
    "--json-schema",
    "--schema-model",
    "--structured-tool-policy",
    "--results",
    "--servers",
    "--auth",
    "--client-metadata-url",
    "--name",
    "--config-path",
    "-c",
    "--shell",
    "--no-shell",
    "-x",
    "--skills",
    "--skills-dir",
    "--agent-cards",
    "--card",
    "--card-tool",
    "--a2a",
    "--a2a-transport",
    "--a2a-oauth",
    "--no-a2a-oauth",
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
    "scaffold",
    "check",
    "cards",
    "plugins",
    "skills",
    "auth",
    "bootstrap",
    "quickstart",
    "--help",
    "-h",
    "--version",
}
