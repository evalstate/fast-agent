from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from fast_agent.command_actions import PluginCommandActionResult, PluginCommandActionSpec
from fast_agent.commands.context import (
    CommandContext,
    NonInteractiveCommandIOBase,
    StaticAgentProvider,
)
from fast_agent.commands.handlers import display as display_handlers
from fast_agent.core.agent_app import AgentRefreshResult
from fast_agent.ui.command_payloads import (
    AgentCommand,
    AttachCommand,
    CommandsCommand,
    HashAgentCommand,
    HistoryFixCommand,
    LoadAgentCardCommand,
    McpConnectCommand,
    McpDisconnectCommand,
    ModelSwitchCommand,
    ShellCommand,
    UnknownCommand,
)
from fast_agent.ui.interactive import command_dispatch
from fast_agent.ui.interactive.command_dispatch import (
    _dispatch_agent_card_payload,
    _dispatch_hash_agent_command,
    _dispatch_history_payload,
    _dispatch_local_ui_payload,
    _dispatch_mcp_connect_command,
    _dispatch_mcp_payload,
    _dispatch_plugin_command_payload,
)
from fast_agent.ui.prompt import parser as prompt_parser

if TYPE_CHECKING:
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.ui.interactive_prompt import InteractivePrompt


class _HistoryOwner:
    def __init__(self, missing_agents: set[str] | None = None) -> None:
        self.missing_agents = missing_agents or set()
        self.checked_agents: list[str] = []

    def _get_agent_or_warn(self, _prompt_provider: object, agent_name: str) -> object | None:
        self.checked_agents.append(agent_name)
        if agent_name in self.missing_agents:
            return None
        return object()


class _RefreshOwner:
    agent_types: dict[str, str] | None = None


class _RefreshProvider:
    def visible_agent_names(self) -> list[str]:
        return ["main", "worker"]

    def visible_agent_types(self, *, force_include: str | None = None) -> dict[str, str]:
        if force_include is None:
            return {"main": "basic"}
        return {force_include: "basic"}


class _WarningRefreshProvider(_RefreshProvider):
    def latest_refresh_result(self) -> AgentRefreshResult:
        return AgentRefreshResult(changed=True, warnings=["Reload warning [card]"])


class _PluginAgent:
    name = "main"
    message_history: list[object] = []
    context = None
    agent_registry = None

    def __init__(self) -> None:
        self.config = SimpleNamespace(
            commands={
                "[plug]": PluginCommandActionSpec(
                    name="[plug]",
                    description="Demo",
                    handler="demo:handler",
                )
            },
            source_path=None,
        )

    def load_message_history(self, messages: list[object] | None) -> None:
        self.message_history = list(messages or [])

    def get_agent(self, name: str) -> object | None:
        del name
        return None

    async def send(self, message: str) -> str:
        return message


class _PluginProvider:
    plugin_commands = None
    plugin_command_base_path = None

    def __init__(self) -> None:
        self.agent = _PluginAgent()

    def get_agent(self, name: str) -> _PluginAgent | None:
        del name
        return self.agent


class _CommandTestIO(NonInteractiveCommandIOBase):
    async def emit(self, message: object) -> None:
        del message


def test_model_value_parser_payloads_have_dispatch_paths() -> None:
    parsed_payload_types = set(prompt_parser._MODEL_VALUE_COMMAND_FACTORIES.values())
    dispatch_payload_types = set(command_dispatch.MODEL_VALUE_COMMAND_HANDLERS) | {
        ModelSwitchCommand
    }

    assert parsed_payload_types == dispatch_payload_types


def test_commands_parser_preserves_discovery_arguments() -> None:
    result = prompt_parser.parse_special_input("/commands skills add --json")

    assert result == CommandsCommand(argument="skills add --json")


@pytest.mark.asyncio
async def test_local_dispatch_sets_shell_command() -> None:
    result = await _dispatch_local_ui_payload(
        ShellCommand("pwd"),
        prompt_provider=cast("AgentApp", object()),
        available_agents_set={"main"},
        agent_name="main",
        buffer_prefill="",
    )

    assert result is not None
    assert result.shell_execute_cmd == "pwd"


@pytest.mark.asyncio
async def test_local_dispatch_appends_attachment_tokens(tmp_path: Path) -> None:
    attachment = tmp_path / "note.txt"
    attachment.write_text("hello")

    result = await _dispatch_local_ui_payload(
        AttachCommand(("note.txt",)),
        prompt_provider=cast("AgentApp", object()),
        available_agents_set={"main"},
        agent_name="main",
        buffer_prefill="inspect",
        shell_working_dir=tmp_path,
    )

    assert result is not None
    assert result.buffer_prefill == f"inspect ^file:{attachment.as_posix()}"


@pytest.mark.asyncio
async def test_local_dispatch_appends_url_attachment_tokens() -> None:
    result = await _dispatch_local_ui_payload(
        AttachCommand(("https://example.test/path?q=1",)),
        prompt_provider=cast("AgentApp", object()),
        available_agents_set={"main"},
        agent_name="main",
        buffer_prefill="inspect",
    )

    assert result is not None
    assert result.buffer_prefill == "inspect ^url:https://example.test/path?q=1"


@pytest.mark.asyncio
async def test_local_dispatch_attachment_error_prints_bracketed_path_literally(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    printed: list[object] = []
    monkeypatch.setattr(command_dispatch, "rich_print", lambda text: printed.append(text))

    result = await _dispatch_local_ui_payload(
        AttachCommand(("[draft].txt",)),
        prompt_provider=cast("AgentApp", object()),
        available_agents_set={"main"},
        agent_name="main",
        buffer_prefill="inspect",
        shell_working_dir=tmp_path,
    )

    assert result is not None
    assert result.buffer_prefill == "inspect"
    assert printed
    assert getattr(printed[0], "plain", "") == "Unable to attach '[draft].txt': [draft].txt"


def test_hash_dispatch_returns_buffer_send_details() -> None:
    result = _dispatch_hash_agent_command(
        HashAgentCommand("worker", "summarize", quiet=True),
        available_agents_set={"worker"},
    )

    assert result.hash_send_target == "worker"
    assert result.hash_send_message == "summarize"
    assert result.hash_send_quiet is True


@pytest.mark.asyncio
async def test_history_dispatch_skips_handler_when_target_agent_missing() -> None:
    owner = _HistoryOwner(missing_agents={"missing"})

    result = await _dispatch_history_payload(
        cast("InteractivePrompt", owner),
        HistoryFixCommand("missing"),
        prompt_provider=cast("AgentApp", object()),
        agent="main",
    )

    assert result is not None
    assert result.handled is True
    assert owner.checked_agents == ["missing"]


@pytest.mark.asyncio
async def test_commands_handler_renders_discovery_markdown() -> None:
    ctx = CommandContext(
        agent_provider=StaticAgentProvider(),
        current_agent_name="main",
        io=_CommandTestIO(),
    )

    outcome = await display_handlers.handle_commands(
        ctx,
        agent_name="main",
        argument="skills add",
    )

    assert len(outcome.messages) == 1
    message = outcome.messages[0]
    assert message.render_markdown is True
    assert message.right_info == "commands"
    assert "# commands skills add" in message.plain_text()


@pytest.mark.asyncio
async def test_mcp_dispatch_handles_missing_server_name_without_context() -> None:
    result = await _dispatch_mcp_payload(
        McpDisconnectCommand(server_name=None, error=None),
        prompt_provider=cast("AgentApp", object()),
        agent="main",
    )

    assert result is not None
    assert result.handled is True


@pytest.mark.asyncio
async def test_mcp_connect_error_prints_bracketed_text_literally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[object] = []
    monkeypatch.setattr(command_dispatch, "rich_print", printed.append)

    def fail_build_command_context(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("parse errors should not build command context")

    monkeypatch.setattr(command_dispatch, "build_command_context", fail_build_command_context)

    result = await _dispatch_mcp_connect_command(
        McpConnectCommand(request=None, error="bad [target]"),
        prompt_provider=cast("AgentApp", object()),
        agent="main",
    )

    assert result.handled is True
    assert [getattr(item, "plain", item) for item in printed] == ["bad [target]"]


def test_mcp_server_error_prints_bracketed_text_literally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[object] = []
    monkeypatch.setattr(command_dispatch, "rich_print", printed.append)

    handler = command_dispatch._mcp_handler(
        McpDisconnectCommand(server_name="demo", error="bad [server]"),
        prompt_provider=cast("AgentApp", object()),
        agent="main",
    )

    assert handler is None
    assert [getattr(item, "plain", item) for item in printed] == ["bad [server]"]


@pytest.mark.asyncio
async def test_agent_card_load_error_prints_bracketed_text_literally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[object] = []
    monkeypatch.setattr(command_dispatch, "rich_print", printed.append)

    result = await _dispatch_agent_card_payload(
        cast("InteractivePrompt", object()),
        LoadAgentCardCommand(
            filename=None,
            add_tool=False,
            remove_tool=False,
            error="bad [card]",
        ),
        prompt_provider=cast("AgentApp", object()),
        agent="main",
        merge_pinned_agents=lambda names: names,
    )

    assert result is not None
    assert result.handled is True
    assert [getattr(item, "plain", item) for item in printed] == ["bad [card]"]


@pytest.mark.asyncio
async def test_agent_command_error_prints_bracketed_text_literally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[object] = []
    monkeypatch.setattr(command_dispatch, "rich_print", printed.append)

    result = await _dispatch_agent_card_payload(
        cast("InteractivePrompt", object()),
        AgentCommand(
            agent_name=None,
            add_tool=False,
            remove_tool=False,
            dump=False,
            error="bad [agent]",
        ),
        prompt_provider=cast("AgentApp", object()),
        agent="main",
        merge_pinned_agents=lambda names: names,
    )

    assert result is not None
    assert result.handled is True
    assert [getattr(item, "plain", item) for item in printed] == ["bad [agent]"]


@pytest.mark.asyncio
async def test_plugin_command_failure_prints_bracketed_text_literally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[object] = []
    monkeypatch.setattr(command_dispatch, "rich_print", printed.append)
    monkeypatch.setattr(command_dispatch, "build_command_context", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(command_dispatch, "_plugin_command_context", lambda **_kwargs: object())

    async def fail_execute(**_kwargs: object) -> None:
        raise RuntimeError("boom [detail]")

    monkeypatch.setattr(command_dispatch, "_execute_plugin_command_action", fail_execute)

    result = await _dispatch_plugin_command_payload(
        cast("InteractivePrompt", _RefreshOwner()),
        UnknownCommand("/[plug] arg"),
        prompt_provider=cast("AgentApp", _PluginProvider()),
        agent="main",
        available_agents_set={"main"},
        merge_pinned_agents=lambda names: names,
        shell_working_dir=None,
    )

    assert result is not None
    assert result.handled is True
    assert [getattr(item, "plain", item) for item in printed] == [
        "Command /[plug] failed: boom [detail]"
    ]


@pytest.mark.asyncio
async def test_plugin_unknown_agent_prints_bracketed_text_literally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[object] = []
    monkeypatch.setattr(command_dispatch, "rich_print", printed.append)
    monkeypatch.setattr(command_dispatch, "build_command_context", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(command_dispatch, "_plugin_command_context", lambda **_kwargs: object())

    async def execute(**_kwargs: object) -> PluginCommandActionResult:
        return PluginCommandActionResult(switch_agent="[ghost]")

    async def emit_outcome(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(command_dispatch, "_execute_plugin_command_action", execute)
    monkeypatch.setattr(command_dispatch, "emit_command_outcome", emit_outcome)

    result = await _dispatch_plugin_command_payload(
        cast("InteractivePrompt", _RefreshOwner()),
        UnknownCommand("/[plug] arg"),
        prompt_provider=cast("AgentApp", _PluginProvider()),
        agent="main",
        available_agents_set={"main"},
        merge_pinned_agents=lambda names: names,
        shell_working_dir=None,
    )

    assert result is not None
    assert result.next_agent is None
    assert [getattr(item, "plain", item) for item in printed] == ["Unknown agent: [ghost]"]


def test_apply_refresh_preferences_prints_warnings_literally(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[object] = []
    monkeypatch.setattr(command_dispatch, "rich_print", printed.append)

    preferred = command_dispatch._apply_refresh_preferences(
        prompt_provider=cast("AgentApp", _WarningRefreshProvider()),
        current_agent="main",
        next_available_agents=["main", "worker"],
        next_available_agents_set={"main", "worker"},
    )

    assert preferred is None
    assert [getattr(item, "plain", item) for item in printed] == ["Reload warning [card]"]
