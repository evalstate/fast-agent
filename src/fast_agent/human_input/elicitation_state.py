"""Simple state management for elicitation Cancel All functionality."""



class ElicitationState:
    """Manages global state for elicitation requests, including disabled servers."""

    def __init__(self):
        self.disabled_servers: set[str] = set()
        self.active_servers: set[str] = set()

    def disable_server(self, server_name: str) -> None:
        """Disable elicitation requests for a specific server."""
        self.disabled_servers.add(server_name)

    def is_disabled(self, server_name: str) -> bool:
        """Check if elicitation is disabled for a server."""
        return server_name in self.disabled_servers

    def start_elicitation(self, server_name: str) -> None:
        """Mark elicitation as active for a server."""
        self.active_servers.add(server_name)

    def end_elicitation(self, server_name: str) -> None:
        """Clear active elicitation state for a server."""
        self.active_servers.discard(server_name)

    def is_active(self, server_name: str) -> bool:
        """Check if elicitation is active for a server."""
        return server_name in self.active_servers

    def enable_server(self, server_name: str) -> None:
        """Re-enable elicitation requests for a specific server."""
        self.disabled_servers.discard(server_name)

    def clear_all(self) -> None:
        """Clear all disabled and active servers."""
        self.disabled_servers.clear()
        self.active_servers.clear()

    def get_disabled_servers(self) -> set[str]:
        """Get a copy of all disabled servers."""
        return self.disabled_servers.copy()


# Global instance for session-scoped Cancel All functionality
elicitation_state = ElicitationState()
