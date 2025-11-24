"""Console entrypoint that forwards to fast-agent's ACP command."""

from fast_agent.cli.commands import acp as fast_agent_acp


def main() -> None:
    """Run the upstream ACP Typer application."""
    fast_agent_acp.main()


if __name__ == "__main__":
    main()
