from __future__ import annotations

from typing import Any

from acp.exceptions import RequestError
from acp.interfaces import Client
from acp.schema import (
    AllowedOutcome,
    DeniedOutcome,
    ReadTextFileRequest,
    ReadTextFileResponse,
    RequestPermissionRequest,
    RequestPermissionResponse,
    SessionNotification,
    WriteTextFileRequest,
    WriteTextFileResponse,
)


class TestClient(Client):
    """
    Minimal ACP client implementation for integration tests.

    This mirrors the helper shipped in agent-client-protocol's own test suite
    and captures notifications, permission decisions, file operations, and
    custom extension calls so tests can assert on the agent's behaviour.
    """

    __test__ = False  # Prevent pytest from treating this as a test case

    def __init__(self) -> None:
        self.permission_outcomes: list[RequestPermissionResponse] = []
        self.files: dict[str, str] = {}
        self.notifications: list[SessionNotification] = []
        self.ext_calls: list[tuple[str, dict[str, Any]]] = []
        self.ext_notes: list[tuple[str, dict[str, Any]]] = []
        self.terminals: dict[str, dict[str, Any]] = {}
        self._terminal_count: int = 0  # For generating terminal IDs like real clients

    def queue_permission_cancelled(self) -> None:
        self.permission_outcomes.append(
            RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        )

    def queue_permission_selected(self, option_id: str) -> None:
        self.permission_outcomes.append(
            RequestPermissionResponse(
                outcome=AllowedOutcome(optionId=option_id, outcome="selected")
            )
        )

    async def requestPermission(
        self, params: RequestPermissionRequest
    ) -> RequestPermissionResponse:
        if self.permission_outcomes:
            return self.permission_outcomes.pop()
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def writeTextFile(self, params: WriteTextFileRequest) -> WriteTextFileResponse:
        self.files[str(params.path)] = params.content
        return WriteTextFileResponse()

    async def readTextFile(self, params: ReadTextFileRequest) -> ReadTextFileResponse:
        content = self.files.get(str(params.path), "default content")
        return ReadTextFileResponse(content=content)

    async def sessionUpdate(self, params: SessionNotification) -> None:
        self.notifications.append(params)

    # Terminal support - implement simple in-memory simulation
    async def terminal_create(self, params: dict[str, Any]) -> dict[str, Any]:
        """Simulate terminal creation and command execution.

        Per ACP spec: CLIENT creates the terminal ID, not the agent.
        This matches how real clients like Toad work (terminal-1, terminal-2, etc.).

        Params per spec: sessionId (required), command (required), args, env, cwd, outputByteLimit (optional)
        Note: sessionId is optional here to support unit tests that call this directly
        """
        session_id = params.get("sessionId", "test-session")  # Required per ACP spec, optional for unit tests
        command = params["command"]
        args = params.get("args", [])
        env = params.get("env", [])  # ACP spec expects array of {name, value} objects
        cwd = params.get("cwd")

        # Validate env format per ACP spec
        if env:
            if not isinstance(env, list):
                raise ValueError(f"env must be an array, got {type(env).__name__}")
            for item in env:
                if not isinstance(item, dict) or "name" not in item or "value" not in item:
                    raise ValueError(f"env items must have 'name' and 'value' keys, got {item}")

        # Generate terminal ID like real clients do (terminal-1, terminal-2, etc.)
        self._terminal_count += 1
        terminal_id = f"terminal-{self._terminal_count}"

        # Build full command if args provided
        full_command = command
        if args:
            full_command = f"{command} {' '.join(args)}"

        # Store terminal state
        self.terminals[terminal_id] = {
            "session_id": session_id,
            "command": full_command,
            "output": f"Executed: {full_command}\nMock output for testing",
            "exit_code": 0,
            "completed": True,
            "env": env,
            "cwd": cwd,
        }

        # Return the ID we created
        return {"terminalId": terminal_id}

    async def terminal_output(self, params: dict[str, Any]) -> dict[str, Any]:
        """Get terminal output.

        Params per spec: sessionId (required), terminalId (required)
        Note: sessionId is optional here to support unit tests that call this directly
        """
        session_id = params.get("sessionId")  # Optional for unit tests
        terminal_id = params["terminalId"]
        terminal = self.terminals.get(terminal_id, {})

        return {
            "output": terminal.get("output", ""),
            "truncated": False,
            "exitCode": terminal.get("exit_code") if terminal.get("completed") else None,
        }

    async def terminal_release(self, params: dict[str, Any]) -> dict[str, Any]:
        """Release terminal resources.

        Params per spec: sessionId (required), terminalId (required)
        Note: sessionId is optional here to support unit tests that call this directly
        """
        session_id = params.get("sessionId")  # Optional for unit tests
        terminal_id = params["terminalId"]
        if terminal_id in self.terminals:
            del self.terminals[terminal_id]
        return {}

    async def terminal_wait_for_exit(self, params: dict[str, Any]) -> dict[str, Any]:
        """Wait for terminal to exit (immediate in simulation).

        Params per spec: sessionId (required), terminalId (required)
        Note: sessionId is optional here to support unit tests that call this directly
        """
        session_id = params.get("sessionId")  # Optional for unit tests
        terminal_id = params["terminalId"]
        terminal = self.terminals.get(terminal_id, {})

        return {
            "exitCode": terminal.get("exit_code", -1),
            "signal": None,
        }

    async def terminal_kill(self, params: dict[str, Any]) -> dict[str, Any]:
        """Kill a running terminal.

        Params per spec: sessionId (required), terminalId (required)
        Note: sessionId is optional here to support unit tests that call this directly
        """
        session_id = params.get("sessionId")  # Optional for unit tests
        terminal_id = params["terminalId"]
        if terminal_id in self.terminals:
            self.terminals[terminal_id]["exit_code"] = -1
            self.terminals[terminal_id]["completed"] = True
        return {}

    async def extMethod(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.ext_calls.append((method, params))
        if method == "example.com/ping":
            return {"response": "pong", "params": params}
        raise RequestError.method_not_found(method)

    async def extNotification(self, method: str, params: dict[str, Any]) -> None:
        self.ext_notes.append((method, params))
