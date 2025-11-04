# Import and re-export AgentMCPServer to avoid circular imports
from fast_agent.mcp.server.agent_server import AgentMCPServer

__all__ = ["AgentMCPServer"]
