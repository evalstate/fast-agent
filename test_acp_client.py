#!/usr/bin/env python3
"""
Simple ACP client to test the fast-agent ACP server.

This client spawns the fast-agent ACP server as a subprocess and
communicates with it using the Agent Client Protocol over stdio.
"""

import asyncio
import sys
from pathlib import Path

from acp import ClientSideConnection
from acp import InitializeRequest, NewSessionRequest, PromptRequest
from acp.helpers import text_block
from acp.stdio import spawn_agent_process
from acp.schema import ClientCapabilities, ClientInfo


async def test_acp_server():
    """Test the ACP server by sending a simple prompt."""
    print("Starting ACP client test...")
    print("=" * 60)

    # Define a simple Client implementation
    # The Client interface receives callbacks from the Agent
    class SimpleClient:
        def __init__(self, conn):
            self.conn = conn

        async def sessionUpdate(self, params):
            """Handle session update notifications from the agent."""
            print(f"Session update: {params}")

        async def requestPermission(self, params):
            """Handle permission requests from the agent."""
            print(f"Permission request: {params}")
            # Auto-approve for testing
            return {"allowed": True}

        async def readTextFile(self, params):
            """Handle file read requests."""
            print(f"Read file request: {params}")
            return {"content": ""}

        async def writeTextFile(self, params):
            """Handle file write requests."""
            print(f"Write file request: {params}")
            return {}

        async def createTerminal(self, params):
            """Handle terminal creation requests."""
            print(f"Create terminal request: {params}")
            return {"terminalId": "test-terminal"}

        async def killTerminal(self, params):
            """Handle terminal kill requests."""
            print(f"Kill terminal request: {params}")
            return {}

        async def releaseTerminal(self, params):
            """Handle terminal release requests."""
            print(f"Release terminal request: {params}")
            return {}

        async def terminalOutput(self, params):
            """Handle terminal output notifications."""
            print(f"Terminal output: {params}")

        async def waitForTerminalExit(self, params):
            """Handle terminal exit wait requests."""
            print(f"Wait for terminal exit: {params}")
            return {"exitCode": 0}

    # Spawn the fast-agent ACP server as a subprocess
    # Note: You'll need to replace this with the actual command to run fast-agent in ACP mode
    fast_agent_command = sys.executable  # Use the same Python interpreter
    fast_agent_args = [
        "-m", "fast_agent.cli",
        "serve",
        "--transport", "acp",
        "--instruction", "You are a helpful assistant. Answer questions concisely.",
    ]

    print(f"Spawning agent: {fast_agent_command} {' '.join(fast_agent_args)}")
    print()

    try:
        async with spawn_agent_process(
            lambda agent: SimpleClient(agent),
            fast_agent_command,
            *fast_agent_args,
        ) as (connection, process):
            print("✓ Agent process spawned successfully")
            print()

            # Step 1: Initialize the connection
            print("Step 1: Initializing connection...")
            init_request = InitializeRequest(
                protocolVersion=1,
                clientCapabilities=ClientCapabilities(
                    fs={"readTextFile": False, "writeTextFile": False},
                    terminal=False,
                ),
                clientInfo=ClientInfo(
                    name="test-acp-client",
                    version="0.1.0",
                ),
            )

            init_response = await connection.initialize(init_request)
            print(f"✓ Initialized successfully")
            print(f"  Agent: {init_response.agentInfo.name} v{init_response.agentInfo.version}")
            print(f"  Protocol version: {init_response.protocolVersion}")
            print(f"  Capabilities: {init_response.agentCapabilities}")
            print()

            # Step 2: Create a new session
            print("Step 2: Creating new session...")
            session_request = NewSessionRequest(
                mcpServers=[],  # No MCP servers for this test
            )

            session_response = await connection.newSession(session_request)
            session_id = session_response.sessionId
            print(f"✓ Session created: {session_id}")
            print()

            # Step 3: Send a prompt
            print("Step 3: Sending prompt...")
            prompt_request = PromptRequest(
                sessionId=session_id,
                prompt=[
                    text_block("What is 2+2? Answer in one sentence."),
                ],
            )

            prompt_response = await connection.prompt(prompt_request)
            print(f"✓ Prompt completed")
            print(f"  Stop reason: {prompt_response.stopReason}")
            print()

            print("=" * 60)
            print("✓ All tests passed!")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("ACP Client Test for fast-agent")
    print()
    asyncio.run(test_acp_server())
