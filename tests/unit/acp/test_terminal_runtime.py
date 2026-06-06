"""Unit tests for ACPTerminalRuntime."""

import os
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.types import TextContent

from fast_agent.acp.terminal_runtime import ACPTerminalRuntime
from fast_agent.constants import DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT
from fast_agent.mcp.tool_execution_handler import NoOpToolExecutionHandler
from fast_agent.mcp.tool_permission_handler import ToolPermissionResult


class RecordingConnection:
    """Simple async connection that records requests and returns preset responses."""

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    async def send_request(self, method: str, params: dict | None = None) -> dict:
        self.calls.append((method, params or {}))
        if self._responses:
            return self._responses.pop(0)
        return {}


class RecordingToolHandler(NoOpToolExecutionHandler):
    def __init__(self) -> None:
        self.starts: list[tuple[str, str, dict | None, str | None]] = []
        self.completes: list[tuple[str, bool, list[Any] | None, str | None]] = []
        self.ensured: list[tuple[str, str, str, dict | None]] = []
        self.denials: list[tuple[str, str, str | None, str]] = []

    async def ensure_tool_call_exists(
        self,
        tool_use_id: str,
        tool_name: str,
        server_name: str,
        arguments: dict | None = None,
    ) -> str:
        self.ensured.append((tool_use_id, tool_name, server_name, arguments))
        return "tool-call-1"

    async def on_tool_start(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict | None,
        tool_use_id: str | None = None,
    ) -> str:
        self.starts.append((tool_name, server_name, arguments, tool_use_id))
        return "tool-call-1"

    async def on_tool_complete(
        self,
        tool_call_id: str,
        success: bool,
        content: list[Any] | None,
        error: str | None,
    ) -> None:
        self.completes.append((tool_call_id, success, content, error))

    async def on_tool_permission_denied(
        self,
        tool_name: str,
        server_name: str,
        tool_use_id: str | None,
        error: str | None = None,
    ) -> None:
        self.denials.append((tool_name, server_name, tool_use_id, error or ""))


class DenyingPermissionHandler:
    async def check_permission(
        self,
        tool_name: str,
        server_name: str,
        arguments: dict[str, Any] | None = None,
        tool_use_id: str | None = None,
    ) -> ToolPermissionResult:
        del tool_name, server_name, arguments, tool_use_id
        return ToolPermissionResult.deny("terminal denied")


def build_runtime(
    responses: list[dict],
    session_id: str = "test-session",
    default_limit: int | None = None,
    tool_handler: RecordingToolHandler | None = None,
    permission_handler: Any | None = None,
):
    """Create a runtime wired to a recording connection."""
    conn = RecordingConnection(responses)
    runtime = ACPTerminalRuntime(
        connection=SimpleNamespace(_conn=conn),
        session_id=session_id,
        activation_reason="test",
        timeout_seconds=90,
        tool_handler=tool_handler,
        default_output_byte_limit=default_limit
        if default_limit is not None
        else DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT,
        permission_handler=permission_handler,
    )
    return runtime, conn


def assert_shell_wrapped(create_params: dict[str, Any], command: str) -> None:
    if os.name == "nt":
        expected_shell = os.environ.get("COMSPEC", "cmd.exe").strip() or "cmd.exe"
        assert create_params["command"] == expected_shell
        assert create_params["args"] == ["/d", "/s", "/c", command]
    else:
        assert create_params["command"] == "/bin/sh"
        assert create_params["args"] == ["-lc", command]


@pytest.mark.asyncio
async def test_env_parameter_transformation_from_object():
    """Test that env parameter is transformed from object to array format."""
    runtime, conn = build_runtime(
        responses=[
            {"terminalId": "terminal-1"},  # terminal/create
            {"exitCode": 0, "signal": None},  # terminal/wait_for_exit
            {"output": "test output", "truncated": False, "exitCode": 0},  # terminal/output
            {},  # terminal/release
        ]
    )

    # Execute command with env as object (LLM-friendly format)
    arguments = {
        "command": "env",
        "env": {
            "PATH": "/usr/local/bin",
            "HOME": "/home/testuser",
            "CUSTOM_VAR": "test123",
        },
    }

    await runtime.execute(arguments)

    # Verify terminal/create was called with transformed env (array format)
    method, create_params = conn.calls[0]
    assert method == "terminal/create"

    # Check that env was transformed to array format
    assert "env" in create_params
    assert isinstance(create_params["env"], list)
    assert len(create_params["env"]) == 3

    # Verify each env item has name and value
    env_dict = {item["name"]: item["value"] for item in create_params["env"]}
    assert env_dict["PATH"] == "/usr/local/bin"
    assert env_dict["HOME"] == "/home/testuser"
    assert env_dict["CUSTOM_VAR"] == "test123"


@pytest.mark.asyncio
async def test_env_parameter_array_is_canonicalized():
    """Test that env parameter in array format is normalized before forwarding."""
    runtime, conn = build_runtime(
        responses=[
            {"terminalId": "terminal-1"},  # terminal/create
            {"exitCode": 0, "signal": None},  # terminal/wait_for_exit
            {"output": "test output", "truncated": False, "exitCode": 0},  # terminal/output
            {},  # terminal/release
        ]
    )

    # Execute command with env already in array format
    env_array = [
        {"name": "PATH", "value": "/usr/local/bin", "extra": "ignored"},
        {"name": "HOME", "value": "/home/testuser"},
    ]
    arguments = {
        "command": "env",
        "env": env_array,
    }

    await runtime.execute(arguments)

    # Verify terminal/create was called with canonical env objects
    _, create_params = conn.calls[0]

    assert create_params["env"] == [
        {"name": "PATH", "value": "/usr/local/bin"},
        {"name": "HOME", "value": "/home/testuser"},
    ]


@pytest.mark.asyncio
async def test_optional_parameters_passed_correctly():
    """Test that all optional parameters are passed to terminal/create."""
    runtime, conn = build_runtime(
        responses=[
            {"terminalId": "terminal-1"},
            {"exitCode": 0, "signal": None},
            {"output": "test output", "truncated": False, "exitCode": 0},
            {},
        ]
    )

    # Execute command with all optional parameters
    arguments = {
        "command": "ls",
        "args": ["-la", "/tmp"],
        "env": {"DEBUG": "true"},
        "cwd": "/home/testuser",
        "outputByteLimit": 10000,
    }

    await runtime.execute(arguments)

    # Verify all parameters were passed to terminal/create
    _, create_params = conn.calls[0]

    assert create_params["command"] == "ls"
    assert create_params["args"] == ["-la", "/tmp"]
    assert create_params["env"] == [{"name": "DEBUG", "value": "true"}]
    assert create_params["cwd"] == "/home/testuser"
    assert create_params["outputByteLimit"] == 10000


@pytest.mark.asyncio
async def test_cwd_parameter_is_trimmed() -> None:
    runtime, conn = build_runtime(
        responses=[
            {"terminalId": "terminal-1"},
            {"exitCode": 0, "signal": None},
            {"output": "test output", "truncated": False, "exitCode": 0},
            {},
        ]
    )

    await runtime.execute({"command": "pwd", "cwd": "  /home/testuser  "})

    _, create_params = conn.calls[0]
    assert create_params["cwd"] == "/home/testuser"


@pytest.mark.asyncio
async def test_command_string_is_split_into_structured_acp_command_and_args():
    runtime, conn = build_runtime(
        responses=[
            {"terminalId": "terminal-1"},
            {"exitCode": 0, "signal": None},
            {"output": "ok", "truncated": False, "exitCode": 0},
            {},
        ]
    )

    await runtime.execute({"command": "git status --short"})

    _, create_params = conn.calls[0]
    assert create_params["command"] == "git"
    assert create_params["args"] == ["status", "--short"]


@pytest.mark.asyncio
async def test_explicit_empty_args_preserves_literal_command() -> None:
    runtime, conn = build_runtime(
        responses=[
            {"terminalId": "terminal-1"},
            {"exitCode": 0, "signal": None},
            {"output": "ok", "truncated": False, "exitCode": 0},
            {},
        ]
    )

    await runtime.execute({"command": "/tmp/my tool", "args": []})

    _, create_params = conn.calls[0]
    assert create_params["command"] == "/tmp/my tool"
    assert "args" not in create_params


@pytest.mark.asyncio
async def test_shell_syntax_is_wrapped_for_acp_terminal_requests():
    runtime, conn = build_runtime(
        responses=[
            {"terminalId": "terminal-1"},
            {"exitCode": 0, "signal": None},
            {"output": "ok", "truncated": False, "exitCode": 0},
            {},
        ]
    )

    command = 'grep -R "Provider.AZURE\\|max_completion_tokens\\|max_tokens" -n src/fast_agent | head -n 80'
    await runtime.execute({"command": command})

    _, create_params = conn.calls[0]
    assert_shell_wrapped(create_params, command)


@pytest.mark.asyncio
async def test_quoted_shell_metacharacters_do_not_force_shell_wrapper() -> None:
    runtime, conn = build_runtime(
        responses=[
            {"terminalId": "terminal-1"},
            {"exitCode": 0, "signal": None},
            {"output": "ok", "truncated": False, "exitCode": 0},
            {},
        ]
    )

    await runtime.execute({"command": 'python -c "print(1)"'})

    _, create_params = conn.calls[0]
    assert create_params["command"] == "python"
    assert create_params["args"] == ["-c", "print(1)"]


@pytest.mark.asyncio
async def test_unquoted_shell_operator_still_forces_shell_wrapper() -> None:
    runtime, conn = build_runtime(
        responses=[
            {"terminalId": "terminal-1"},
            {"exitCode": 0, "signal": None},
            {"output": "ok", "truncated": False, "exitCode": 0},
            {},
        ]
    )

    command = "echo one && echo two"
    await runtime.execute({"command": command})

    _, create_params = conn.calls[0]
    assert_shell_wrapped(create_params, command)


@pytest.mark.parametrize(
    "command",
    [
        "echo $HOME",
        "ls *.py",
        "cd /tmp",
        "export FOO=bar",
        "FOO=bar python script.py",
        "echo `pwd`",
        "echo $(pwd)",
        "ls ~/work",
        "sleep 1 &",
        "echo primary || echo fallback",
    ],
)
@pytest.mark.asyncio
async def test_common_shell_syntax_uses_shell_wrapper(command: str) -> None:
    runtime, conn = build_runtime(
        responses=[
            {"terminalId": "terminal-1"},
            {"exitCode": 0, "signal": None},
            {"output": "ok", "truncated": False, "exitCode": 0},
            {},
        ]
    )

    await runtime.execute({"command": command})

    _, create_params = conn.calls[0]
    assert_shell_wrapped(create_params, command)


@pytest.mark.asyncio
async def test_explicit_args_bypass_shell_wrapper_for_shell_like_text() -> None:
    runtime, conn = build_runtime(
        responses=[
            {"terminalId": "terminal-1"},
            {"exitCode": 0, "signal": None},
            {"output": "ok", "truncated": False, "exitCode": 0},
            {},
        ]
    )

    await runtime.execute({"command": "echo", "args": ["$HOME"]})

    _, create_params = conn.calls[0]
    assert create_params["command"] == "echo"
    assert create_params["args"] == ["$HOME"]


@pytest.mark.asyncio
async def test_default_output_byte_limit_used_when_missing():
    """Ensure a sensible default output limit is applied."""
    runtime, conn = build_runtime(
        responses=[
            {"terminalId": "terminal-1"},
            {"exitCode": 0, "signal": None},
            {"output": "test output", "truncated": False, "exitCode": 0},
            {},
        ]
    )

    await runtime.execute({"command": "echo default"})

    _, create_params = conn.calls[0]
    assert create_params["outputByteLimit"] == DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT


@pytest.mark.asyncio
async def test_custom_default_output_byte_limit_overrides_baseline():
    """Verify callers can override the default terminal output limit."""
    custom_limit = 12345
    runtime, conn = build_runtime(
        responses=[
            {"terminalId": "terminal-override"},
            {"exitCode": 0, "signal": None},
            {"output": "test output", "truncated": False, "exitCode": 0},
            {},
        ],
        default_limit=custom_limit,
    )

    await runtime.execute({"command": "echo custom"})

    _, create_params = conn.calls[0]
    assert create_params["outputByteLimit"] == custom_limit


@pytest.mark.asyncio
async def test_boolean_output_byte_limit_falls_back_to_default() -> None:
    runtime, conn = build_runtime(
        responses=[
            {"terminalId": "terminal-1"},
            {"exitCode": 0, "signal": None},
            {"output": "test output", "truncated": False, "exitCode": 0},
            {},
        ]
    )

    await runtime.execute({"command": "echo default", "outputByteLimit": True})

    _, create_params = conn.calls[0]
    assert create_params["outputByteLimit"] == DEFAULT_TERMINAL_OUTPUT_BYTE_LIMIT


@pytest.mark.asyncio
async def test_execute_rejects_invalid_argument_payloads() -> None:
    runtime, conn = build_runtime(responses=[])

    non_dict_result = await runtime.execute(None)  # type: ignore[arg-type]
    missing_command_result = await runtime.execute({})
    blank_command_result = await runtime.execute({"command": "   "})

    assert non_dict_result.isError is True
    assert missing_command_result.isError is True
    assert blank_command_result.isError is True
    assert conn.calls == []
    assert non_dict_result.content is not None
    assert missing_command_result.content is not None
    assert blank_command_result.content is not None
    assert isinstance(non_dict_result.content[0], TextContent)
    assert isinstance(missing_command_result.content[0], TextContent)
    assert isinstance(blank_command_result.content[0], TextContent)
    assert non_dict_result.content[0].text == "Error: arguments must be a dict"
    assert (
        missing_command_result.content[0].text
        == "Error: 'command' argument is required and must be a string"
    )
    assert (
        blank_command_result.content[0].text
        == "Error: 'command' argument is required and must be a string"
    )


@pytest.mark.asyncio
async def test_execute_rejects_invalid_args_before_terminal_create() -> None:
    tool_handler = RecordingToolHandler()
    runtime, conn = build_runtime(responses=[], tool_handler=tool_handler)

    result = await runtime.execute({"command": "echo", "args": "abc"})

    assert result.isError is True
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "Error: 'args' argument must be a list of strings"
    assert conn.calls == []
    assert tool_handler.starts == []
    assert tool_handler.completes == []


@pytest.mark.asyncio
async def test_execute_rejects_invalid_env_before_terminal_create() -> None:
    tool_handler = RecordingToolHandler()
    runtime, conn = build_runtime(responses=[], tool_handler=tool_handler)

    result = await runtime.execute({"command": "env", "env": {"RETRIES": 3}})

    assert result.isError is True
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "Error: 'env' argument must contain string keys and values"
    assert conn.calls == []
    assert tool_handler.starts == []
    assert tool_handler.completes == []


@pytest.mark.asyncio
async def test_execute_rejects_empty_string_env_before_terminal_create() -> None:
    tool_handler = RecordingToolHandler()
    runtime, conn = build_runtime(responses=[], tool_handler=tool_handler)

    result = await runtime.execute({"command": "env", "env": ""})

    assert result.isError is True
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert (
        result.content[0].text
        == "Error: 'env' argument must be an object with string keys and values"
    )
    assert conn.calls == []
    assert tool_handler.starts == []
    assert tool_handler.completes == []


@pytest.mark.asyncio
async def test_execute_rejects_invalid_cwd_before_terminal_create() -> None:
    tool_handler = RecordingToolHandler()
    runtime, conn = build_runtime(responses=[], tool_handler=tool_handler)

    result = await runtime.execute({"command": "pwd", "cwd": 123})

    assert result.isError is True
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "Error: 'cwd' argument must be a string"
    assert conn.calls == []
    assert tool_handler.starts == []
    assert tool_handler.completes == []


@pytest.mark.asyncio
async def test_execute_rejects_blank_cwd_before_terminal_create() -> None:
    tool_handler = RecordingToolHandler()
    runtime, conn = build_runtime(responses=[], tool_handler=tool_handler)

    result = await runtime.execute({"command": "pwd", "cwd": "   "})

    assert result.isError is True
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    assert result.content[0].text == "Error: 'cwd' argument must be a non-empty string"
    assert conn.calls == []
    assert tool_handler.starts == []
    assert tool_handler.completes == []


@pytest.mark.asyncio
async def test_permission_denial_notifies_tool_progress() -> None:
    tool_handler = RecordingToolHandler()
    runtime, conn = build_runtime(
        responses=[],
        tool_handler=tool_handler,
        permission_handler=DenyingPermissionHandler(),
    )

    result = await runtime.execute({"command": "echo denied"}, tool_use_id="llm-tool-1")

    assert result.isError is True
    assert conn.calls == []
    assert tool_handler.starts == []
    assert tool_handler.completes == []
    assert tool_handler.ensured == [
        (
            "llm-tool-1",
            "execute",
            "acp_terminal",
            {"command": "echo denied"},
        )
    ]
    assert tool_handler.denials == [("execute", "acp_terminal", "llm-tool-1", "terminal denied")]


@pytest.mark.asyncio
async def test_truncated_output_includes_limit_context() -> None:
    custom_limit = 12000
    runtime, _ = build_runtime(
        responses=[
            {"terminalId": "terminal-truncated"},
            {"exitCode": 0, "signal": None},
            {"output": "partial output", "truncated": True, "exitCode": 0},
            {},
        ],
        default_limit=custom_limit,
    )

    result = await runtime.execute({"command": "echo test"})

    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    text = result.content[0].text
    assert "[Output truncated by ACP terminal outputByteLimit" in text
    assert "12000 bytes" in text


@pytest.mark.asyncio
async def test_malformed_terminal_output_is_normalized() -> None:
    runtime, _ = build_runtime(
        responses=[
            {"terminalId": "terminal-malformed-output"},
            {"exitCode": 0, "signal": None},
            {"output": None, "truncated": True, "exitCode": 0},
            {},
        ],
        default_limit=8192,
    )

    result = await runtime.execute({"command": "echo test"})

    assert result.isError is False
    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    text = result.content[0].text
    assert "[Output truncated by ACP terminal outputByteLimit" in text
    assert "8192 bytes" in text
    assert text.endswith("[Exit code: 0]")


@pytest.mark.asyncio
async def test_string_truncated_flag_is_not_treated_as_true() -> None:
    runtime, _ = build_runtime(
        responses=[
            {"terminalId": "terminal-string-truncated"},
            {"exitCode": 0, "signal": None},
            {"output": "complete output", "truncated": "false", "exitCode": 0},
            {},
        ]
    )

    result = await runtime.execute({"command": "echo test"})

    assert result.content is not None
    assert isinstance(result.content[0], TextContent)
    text = result.content[0].text
    assert "[Output truncated by ACP terminal outputByteLimit" not in text
    assert text == "complete output\n\n[Exit code: 0]"


@pytest.mark.asyncio
async def test_missing_terminal_id_completes_tool_progress() -> None:
    tool_handler = RecordingToolHandler()
    runtime, conn = build_runtime(
        responses=[{}],
        tool_handler=tool_handler,
    )

    result = await runtime.execute({"command": "echo test"}, tool_use_id="llm-tool-1")

    assert result.isError is True
    assert [method for method, _params in conn.calls] == ["terminal/create"]
    assert tool_handler.starts == [
        ("execute", "acp_terminal", {"command": "echo test"}, "llm-tool-1")
    ]
    assert tool_handler.completes == [
        (
            "tool-call-1",
            False,
            None,
            "Error: Client did not return terminal ID",
        )
    ]


@pytest.mark.asyncio
async def test_session_id_in_all_terminal_requests():
    """Test that sessionId IS included in all terminal method parameters (per ACP spec)."""
    runtime, conn = build_runtime(
        responses=[
            {"terminalId": "terminal-1"},
            {"exitCode": 0, "signal": None},
            {"output": "test output", "truncated": False, "exitCode": 0},
            {},
        ],
        session_id="test-session-123",
    )

    await runtime.execute({"command": "echo test"})

    # Verify sessionId IS in all terminal method parameters (per ACP spec)
    # The ACP specification requires sessionId in all terminal methods
    expected_calls = [
        ("terminal/create", True),
        ("terminal/wait_for_exit", True),
        ("terminal/output", True),
        ("terminal/release", True),
    ]

    assert len(conn.calls) == 4

    for (expected_method, should_have_session), (method_name, params) in zip(
        expected_calls, conn.calls
    ):
        assert method_name == expected_method
        if should_have_session:
            assert "sessionId" in params
            assert params["sessionId"] == "test-session-123"


# Note: Timeout handling is difficult to test reliably in unit tests due to
# asyncio.wait_for() behavior with mocks. The timeout path is tested manually and
# the code includes proper sessionId in all cleanup calls (kill, output, release).
# The test_session_id_in_all_terminal_requests test above verifies sessionId is
# included in the successful path for all terminal methods.
