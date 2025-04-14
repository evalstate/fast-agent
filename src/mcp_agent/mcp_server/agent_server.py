# src/mcp_agent/mcp_server/agent_server.py

import asyncio

from mcp.server.fastmcp import Context as MCPContext
from mcp.server.fastmcp import FastMCP

import mcp_agent
import mcp_agent.core
import mcp_agent.core.prompt
from mcp_agent.core.agent_app import AgentApp


class AgentMCPServer:
    """Exposes FastAgent agents as MCP tools through an MCP server."""

    def __init__(
        self,
        agent_app: AgentApp,
        server_name: str = "FastAgent-MCP-Server",
        server_description: str | None = None,
    ) -> None:
        self.agent_app = agent_app
        self.mcp_server: FastMCP = FastMCP(
            name=server_name,
            instructions=server_description
            or f"This server provides access to {len(agent_app._agents)} agents",
        )
        self.setup_tools()

    def setup_tools(self) -> None:
        """Register all agents as MCP tools."""
        for agent_name, agent in self.agent_app._agents.items():
            self.register_agent_tools(agent_name, agent)

    def register_agent_tools(self, agent_name: str, agent) -> None:
        """Register tools for a specific agent."""

        # Basic send message tool
        @self.mcp_server.tool(
            name=f"{agent_name}_send",
            description=f"Send a message to the {agent_name} agent",
        )
        async def send_message(message: str, ctx: MCPContext) -> str:
            """Send a message to the agent and return its response."""

            # Get the agent's context
            agent_context = getattr(agent, "context", None)

            # Define the function to execute
            async def execute_send():
                return await agent.send(message)

            # Execute with bridged context
            if agent_context and ctx:
                return await self.with_bridged_context(agent_context, ctx, execute_send)
            else:
                return await execute_send()

        # Register a history prompt for this agent
        @self.mcp_server.prompt(
            name=f"{agent_name}_history",
            description=f"Conversation history for the {agent_name} agent",
        )
        async def get_history_prompt() -> list:
            """Return the conversation history as MCP messages."""
            # Get the conversation history from the agent's LLM
            if not hasattr(agent, "_llm") or agent._llm is None:
                return []

            # Convert the multipart message history to standard PromptMessages
            multipart_history = agent._llm.message_history
            prompt_messages = mcp_agent.core.prompt.Prompt.from_multipart(multipart_history)

            # In FastMCP, we need to return the raw list of messages
            # that matches the structure that FastMCP expects (list of dicts with role/content)
            return [{"role": msg.role, "content": msg.content} for msg in prompt_messages]

    def run(self, transport: str = "sse", host: str = "0.0.0.0", port: int = 8000) -> None:
        """Run the MCP server."""
        if transport == "sse":
            # For running as a web server
            self.mcp_server.settings.host = host
            self.mcp_server.settings.port = port

        self.mcp_server.run(transport=transport)

    async def run_async(
        self, transport: str = "sse", host: str = "0.0.0.0", port: int = 8000
    ) -> None:
        """Run the MCP server asynchronously."""
        if transport == "sse":
            self.mcp_server.settings.host = host
            self.mcp_server.settings.port = port
            try:
                await self.mcp_server.run_sse_async()
            except (asyncio.CancelledError, KeyboardInterrupt):
                print("Server Stopped (CTRL+C)")
                return
        else:  # stdio
            try:
                await self.mcp_server.run_stdio_async()
            except (asyncio.CancelledError, KeyboardInterrupt):
                # Gracefully handle cancellation during shutdown
                print("Server Stopped (CTRL+C)")
                return

    async def with_bridged_context(self, agent_context, mcp_context, func, *args, **kwargs):
        """
        Execute a function with bridged context between MCP and agent

        Args:
            agent_context: The agent's context object
            mcp_context: The MCP context from the tool call
            func: The function to execute
            args, kwargs: Arguments to pass to the function
        """
        # Store original progress reporter if it exists
        original_progress_reporter = None
        if hasattr(agent_context, "progress_reporter"):
            original_progress_reporter = agent_context.progress_reporter

        # Store MCP context in agent context for nested calls
        agent_context.mcp_context = mcp_context

        # Create bridged progress reporter
        async def bridged_progress(progress, total=None) -> None:
            if mcp_context:
                await mcp_context.report_progress(progress, total)
            if original_progress_reporter:
                await original_progress_reporter(progress, total)

        # Install bridged progress reporter
        if hasattr(agent_context, "progress_reporter"):
            agent_context.progress_reporter = bridged_progress

        try:
            # Call the function
            return await func(*args, **kwargs)
        finally:
            # Restore original progress reporter
            if hasattr(agent_context, "progress_reporter"):
                agent_context.progress_reporter = original_progress_reporter

            # Remove MCP context reference
            if hasattr(agent_context, "mcp_context"):
                delattr(agent_context, "mcp_context")

    async def shutdown(self):
        """Gracefully shutdown the MCP server and its resources."""
        # Your MCP server may have additional cleanup code here
        pass
