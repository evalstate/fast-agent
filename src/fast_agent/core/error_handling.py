"""
Error handling utilities for agent operations.
"""

import sys

from rich import print


def handle_error(e: Exception, error_type: str, suggestion: str | None = None) -> None:
    """
    Handle errors with consistent formatting and messaging.

    Args:
        e: The exception that was raised
        error_type: Type of error to display
        suggestion: Optional suggestion message to display
    """
    print(f"\n[bold red]{error_type}:", file=sys.stderr)
    print(getattr(e, "message", str(e)), file=sys.stderr)
    if hasattr(e, "details") and e.details:
        print("\nDetails:", file=sys.stderr)
        print(e.details, file=sys.stderr)
    if suggestion:
        print(f"\n{suggestion}", file=sys.stderr)
        print(file=sys.stderr)
        print("Visit https://fast-agent.ai/ for more information", file=sys.stderr)
