"""Interactive-only Copilot activation shared by startup and the TUI picker."""

import os

import typer

from fast_agent.config import CopilotSettings
from fast_agent.core.exceptions import ProviderKeyError, format_fast_agent_error
from fast_agent.llm.provider.copilot.broker import CopilotBroker
from fast_agent.llm.provider.copilot.oauth import login_copilot_oauth_async


async def activate_copilot(settings: CopilotSettings) -> bool:
    """Load credentials or start device login after explicit picker selection."""
    from fast_agent.ui import console

    try:
        broker = CopilotBroker(settings)
        if await broker.has_credentials():
            return True
        if "COPILOT_GITHUB_TOKEN" in os.environ:
            raise ProviderKeyError(
                "Copilot environment credential is unavailable.",
                "Check COPILOT_GITHUB_TOKEN or unset it to use saved OAuth credentials. "
                "Device login will not replace an explicit environment token.",
            )
        console.ensure_blocking_console()
        typer.echo("Starting GitHub Copilot device-code login… (Ctrl+C to cancel)", err=True)
        # Await directly: task cancellation must stop polling before credentials are saved.
        await login_copilot_oauth_async()
        if not await broker.has_credentials():
            raise ProviderKeyError(
                "Copilot credential is unavailable after GitHub login.",
                "Check the fast-agent credential store and retry selection.",
            )
        return True
    except ProviderKeyError as exc:
        typer.echo(format_fast_agent_error(exc), err=True)
        return False
    except (EOFError, KeyboardInterrupt, typer.Abort):
        typer.echo("Copilot login cancelled.", err=True)
        return False
