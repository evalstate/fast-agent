"""
Error handling utilities for agent operations.
"""

from fast_agent.core.exceptions import FastAgentError
from fast_agent.ui.console import error_console


def handle_error(e: Exception, error_type: str, suggestion: str | None = None) -> None:
    """
    Handle errors with consistent formatting and messaging.

    Args:
        e: The exception that was raised
        error_type: Type of error to display
        suggestion: Optional suggestion message to display
    """
    error_console.print(f"\n[bold red]{error_type}:")
    if isinstance(e, FastAgentError):
        error_console.print(e.message)
        details = e.details
    else:
        error_console.print(str(e))
        details = ""
    if details:
        error_console.print("\nDetails:")
        error_console.print(details)
    if suggestion:
        error_console.print(f"\n{suggestion}")
        error_console.print()
        error_console.print("Visit https://fast-agent.ai/ for more information")
