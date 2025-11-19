"""
Centralized progress display configuration for MCP Agent.
Provides a shared progress display instance for consistent progress handling.
"""

from fast_agent.ui.console import console
from fast_agent.ui.rich_progress import RichProgressDisplay

# Main progress display instance - shared across the application
progress_display = RichProgressDisplay(console)
