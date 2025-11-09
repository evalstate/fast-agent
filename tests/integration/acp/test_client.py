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
        # Terminal tracking for tests
        self.terminals: dict[str, dict[str, Any]] = {}

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

    async def requestPermission(self, params: RequestPermissionRequest) -> RequestPermissionResponse:
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

    async def createTerminal(self, params: Any) -> Any:
        """Track terminal creation for test validation."""
        terminal_id = params.get("terminalId", f"test-terminal-{len(self.terminals)}")
        self.terminals[terminal_id] = {
            "command": params.get("command"),
            "args": params.get("args", []),
            "env": params.get("env", {}),
            "cwd": params.get("cwd"),
            "output": "",
            "exitCode": None,
            "released": False,
        }
        return {"terminalId": terminal_id}

    async def terminalOutput(self, params: Any) -> Any:
        """Return tracked terminal output for tests."""
        terminal_id = params.get("terminalId")
        terminal = self.terminals.get(terminal_id, {})
        return {
            "output": terminal.get("output", ""),
            "truncated": False,
            "exitStatus": {"exitCode": terminal.get("exitCode")}
            if terminal.get("exitCode") is not None
            else None,
        }

    async def releaseTerminal(self, params: Any) -> Any:
        """Mark terminal as released for test validation."""
        terminal_id = params.get("terminalId")
        if terminal_id in self.terminals:
            self.terminals[terminal_id]["released"] = True
        return {}

    async def waitForTerminalExit(self, params: Any) -> Any:
        """Return terminal exit status for tests."""
        terminal_id = params.get("terminalId")
        terminal = self.terminals.get(terminal_id, {})
        return {"exitCode": terminal.get("exitCode", 0), "signal": None}

    async def killTerminal(self, params: Any) -> Any:
        """Mark terminal as killed for test validation."""
        terminal_id = params.get("terminalId")
        if terminal_id in self.terminals:
            self.terminals[terminal_id]["exitCode"] = -1
        return {}

    async def extMethod(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.ext_calls.append((method, params))
        if method == "example.com/ping":
            return {"response": "pong", "params": params}
        raise RequestError.method_not_found(method)

    async def extNotification(self, method: str, params: dict[str, Any]) -> None:
        self.ext_notes.append((method, params))
