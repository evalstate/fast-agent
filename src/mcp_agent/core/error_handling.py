"""
Error handling utilities for agent operations.
"""

from rich import print


def handle_error(e: Exception, error_type: str, suggestion: str = None) -> None:
    """
    Handle errors with consistent formatting and messaging.

    Args:
        e: The exception that was raised
        error_type: Type of error to display
        suggestion: Optional suggestion message to display
    """
    print(f"\n[bold red]{error_type}:")
    print(getattr(e, "message", str(e)))
    if hasattr(e, "details") and e.details:
        print("\nDetails:")
        print(e.details)
    if suggestion:
        print(f"\n{suggestion}")
