"""Shared CLI option definitions for fast-agent commands to reduce duplication."""

import typer


class CommonAgentOptions:
    """Shared options for agent commands to reduce duplication."""

    @staticmethod
    def config_path():
        return typer.Option(None, "--config-path", "-c", help="Path to config file")

    @staticmethod
    def instruction():
        return typer.Option(
            None,
            "--instruction",
            "-i",
            help="Path to file or URL containing instruction for the agent",
        )

    @staticmethod
    def servers():
        return typer.Option(None, "--servers", help="Comma-separated list of server names to enable from config")

    @staticmethod
    def agent_cards():
        return typer.Option(
            None,
            "--agent-cards",
            "--card",
            help="Path or URL to an AgentCard file or directory (repeatable)",
        )

    @staticmethod
    def card_tools():
        return typer.Option(
            None,
            "--card-tool",
            help="Path or URL to an AgentCard file or directory to load as tools (repeatable)",
        )

    @staticmethod
    def urls():
        return typer.Option(None, "--url", help="Comma-separated list of HTTP/SSE URLs to connect to")

    @staticmethod
    def auth():
        return typer.Option(
            None,
            "--auth",
            help=(
                "Authorization token value for URL-based servers "
                "(pass token only; optional 'Bearer ' prefix is accepted)"
            ),
        )

    @staticmethod
    def client_metadata_url():
        return typer.Option(
            None,
            "--client-metadata-url",
            help=(
                "OAuth Client ID Metadata Document URL for URL-based servers "
                "(used when server does not support dynamic client registration)"
            ),
        )

    @staticmethod
    def model():
        return typer.Option(None, "--model", "--models", help="Override the default model (e.g., haiku, sonnet, gpt-4)")

    @staticmethod
    def agent():
        return typer.Option(
            None,
            "--agent",
            help="Target a specific agent by name for --message, --prompt-file, and initial interactive mode",
        )
    
    @staticmethod
    def env_dir():
        return typer.Option(None, "--env", help="Override the base fast-agent environment directory")

    @staticmethod
    def noenv():
        return typer.Option(
            False,
            "--noenv",
            "--no-env",
            help="Run without implicit environment-side effects",
        )

    @staticmethod
    def skills_dir():
        return typer.Option(None, "--skills-dir", "--skills", help="Override the default skills directory")

    @staticmethod
    def npx():
        return typer.Option(None, "--npx", help="NPX package and args to run as MCP server (quoted)")

    @staticmethod
    def uvx():
        return typer.Option(None, "--uvx", help="UVX package and args to run as MCP server (quoted)")

    @staticmethod
    def stdio():
        return typer.Option(None, "--stdio", help="Command to run as STDIO MCP server (quoted)")

    @staticmethod
    def shell():
        return typer.Option(False, "--shell", "-x", help="Enable a local shell runtime and expose the execute tool (bash or pwsh).")

    @staticmethod
    def smart():
        return typer.Option(
            False,
            "--smart",
            help="Prefer a smart default agent when fast-agent creates the default agent.",
        )

    @staticmethod
    def reload():
        return typer.Option(False, "--reload", help="Enable manual AgentCard reloads")

    @staticmethod
    def watch():
        return typer.Option(False, "--watch", help="Watch AgentCard paths and reload automatically")
