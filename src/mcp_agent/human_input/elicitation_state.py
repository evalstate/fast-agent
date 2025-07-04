"""Simple state management for elicitation Cancel All functionality."""

from typing import Set


class ElicitationState:
    """Manages global state for elicitation requests, including disabled servers."""
    
    def __init__(self):
        self.disabled_servers: Set[str] = set()
    
    def disable_server(self, server_name: str) -> None:
        """Disable elicitation requests for a specific server."""
        self.disabled_servers.add(server_name)
    
    def is_disabled(self, server_name: str) -> bool:
        """Check if elicitation is disabled for a server."""
        return server_name in self.disabled_servers
    
    def enable_server(self, server_name: str) -> None:
        """Re-enable elicitation requests for a specific server."""
        self.disabled_servers.discard(server_name)
    
    def clear_all(self) -> None:
        """Clear all disabled servers."""
        self.disabled_servers.clear()
    
    def get_disabled_servers(self) -> Set[str]:
        """Get a copy of all disabled servers."""
        return self.disabled_servers.copy()


# Global instance for session-scoped Cancel All functionality
elicitation_state = ElicitationState()