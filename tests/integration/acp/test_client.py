from __future__ import annotations

from typing import Any

from acp.exceptions import RequestError
from acp.interfaces import Client
from acp.schema import (
    AllowedOutcome,
    CreateTerminalResponse,
    DeniedOutcome,
    EnvVariable,
    KillTerminalCommandResponse,
    PermissionOption,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    TerminalExitStatus,
    TerminalOutputResponse,
    ToolCallUpdate,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)


class TestClient(Client):
    """
    Minimal ACP client implementation for integration tests.

    This mirrors the helper shipped in agent-client-protocol's own test suite
    and captures notifications, permission decisions, file operations, and
    custom extension calls so tests can assert on the agent's behaviour.

    Uses the new SDK 0.7.0 snake_case method names with flattened parameters.
    """

    __test__ = False  # Prevent pytest from treating this as a test case

    def __init__(self) -> None:
        self.permission_outcomes: list[RequestPermissionResponse] = []
        self.files: dict[str, str] = {}
        self.notifications: list[dict[str, Any]] = []  # Store as dicts for flexibility
        self.ext_calls: list[tuple[str, dict[str, Any]]] = []
        self.ext_notes: list[tuple[str, dict[str, Any]]] = []
        self.terminals: dict[str, dict[str, Any]] = {}
        self._terminal_count: int = 0  # For generating terminal IDs like real clients

    def reset(self) -> None:
        self.permission_outcomes.clear()
        self.files.clear()
        self.notifications.clear()
        self.ext_calls.clear()
        self.ext_notes.clear()
        self.terminals.clear()
        self._terminal_count = 0

    def queue_permission_cancelled(self) -> None:
        self.permission_outcomes.append(
            RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
        )

    def queue_permission_selected(self, option_id: str) -> None:
        self.permission_outcomes.append(
            RequestPermissionResponse(
                outcome=AllowedOutcome(option_id=option_id, outcome="selected")
            )
        )

    # New SDK 0.7.0 style: snake_case with flattened parameters
    async def request_permission(
        self,
        options: list[PermissionOption],
        session_id: str,
        tool_call: ToolCallUpdate,
        **kwargs: Any,
    ) -> RequestPermissionResponse:
        if self.permission_outcomes:
            return self.permission_outcomes.pop()
        return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))

    async def write_text_file(
        self,
        content: str,
        path: str,
        session_id: str,
        **kwargs: Any,
    ) -> WriteTextFileResponse | None:
        self.files[str(path)] = content
        return WriteTextFileResponse()

    async def read_text_file(
        self,
        path: str,
        session_id: str,
        limit: int | None = None,
        line: int | None = None,
        **kwargs: Any,
    ) -> ReadTextFileResponse:
        content = self.files.get(str(path), "default content")
        return ReadTextFileResponse(content=content)

    async def session_update(
        self,
        session_id: str,
        update: Any,
        **kwargs: Any,
    ) -> None:
        """Capture session updates for assertions."""
        meta = kwargs.get("_meta")
        if meta is None and "field_meta" in kwargs:
            meta = kwargs.get("field_meta")
        self.notifications.append(
            {
                "session_id": session_id,
                "update": update,
                "meta": meta,
            }
        )

    # Terminal support - implement simple in-memory simulation
    async def create_terminal(
        self,
        command: str,
        session_id: str,
        args: list[str] | None = None,
        cwd: str | None = None,
        env: list[EnvVariable] | None = None,
        output_byte_limit: int | None = None,
        **kwargs: Any,
    ) -> CreateTerminalResponse:
        """Simulate terminal creation and command execution.

        Per ACP spec: CLIENT creates the terminal ID, not the agent.
        This matches how real clients like Toad work (terminal-1, terminal-2, etc.).
        """
        # Validate env format per ACP spec
        if env:
            if not isinstance(env, list):
                raise ValueError(f"env must be an array, got {type(env).__name__}")

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
        return CreateTerminalResponse(terminal_id=terminal_id)

    async def terminal_output(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> TerminalOutputResponse:
        """Get terminal output."""
        terminal = self.terminals.get(terminal_id, {})
        exit_code = terminal.get("exit_code")
        if isinstance(exit_code, int) and exit_code >= 0:
            exit_status = TerminalExitStatus(exit_code=exit_code)
        elif isinstance(exit_code, int) and exit_code < 0:
            exit_status = TerminalExitStatus(exit_code=None, signal="SIGKILL")
        else:
            exit_status = None

        return TerminalOutputResponse(
            output=terminal.get("output", ""),
            truncated=False,
            exit_status=exit_status,
        )

    async def release_terminal(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> ReleaseTerminalResponse | None:
        """Release terminal resources."""
        if terminal_id in self.terminals:
            del self.terminals[terminal_id]
        return ReleaseTerminalResponse()

    async def wait_for_terminal_exit(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> WaitForTerminalExitResponse:
        """Wait for terminal to exit (immediate in simulation)."""
        terminal = self.terminals.get(terminal_id, {})
        exit_code = terminal.get("exit_code")
        if isinstance(exit_code, int) and exit_code >= 0:
            return WaitForTerminalExitResponse(exit_code=exit_code, signal=None)

        # Unknown or negative exit -> model as killed/terminated with no exit code
        return WaitForTerminalExitResponse(
            exit_code=None, signal="SIGKILL" if exit_code else None
        )

    async def kill_terminal(
        self,
        session_id: str,
        terminal_id: str,
        **kwargs: Any,
    ) -> KillTerminalCommandResponse | None:
        """Kill a running terminal."""
        if terminal_id in self.terminals:
            self.terminals[terminal_id]["exit_code"] = -1
            self.terminals[terminal_id]["completed"] = True
        return KillTerminalCommandResponse()

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.ext_calls.append((method, params))
        if method == "example.com/ping":
            return {"response": "pong", "params": params}
        raise RequestError.method_not_found(method)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        self.ext_notes.append((method, params))

    def on_connect(self, conn: Any) -> None:
        """Called when connected to agent. No-op for test client."""
        pass
