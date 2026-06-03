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
from fast_agent.commands.results import CommandOutcome
from fast_agent.core.agent_app import AgentRefreshResult
from fast_agent.ui.command_payloads import (
    AgentCommand,
    AttachCommand,
    CardsCommand,
    CheckCommand,
    ClearCommand,
    CommandsCommand,
    HashAgentCommand,
    HistoryFixCommand,
    HistoryRewindCommand,
    ListToolsCommand,
    LoadAgentCardCommand,
    LoadPromptCommand,
    McpConnectCommand,
    McpDisconnectCommand,
    McpListCommand,
    McpReconnectCommand,
    McpSessionCommand,
    ModelsCommand,
    ModelSwitchCommand,
    ModelWebFetchCommand,
    ResumeSessionCommand,
    ShellCommand,
    ShowUsageCommand,
    TitleSessionCommand,
    UnknownCommand,
)
from fast_agent.ui.interactive import command_dispatch
from fast_agent.ui.interactive.command_dispatch import (
    _attachment_token,
    _catalog_handler,
    _dispatch_agent_card_payload,
    _dispatch_display_payload,
    _dispatch_hash_agent_command,
    _dispatch_history_payload,
    _dispatch_local_ui_payload,
    _dispatch_mcp_connect_command,
    _dispatch_mcp_payload,
    _dispatch_plugin_command_payload,
    _dispatch_prompt_payload,
    _dispatch_session_payload,
    _DispatchStep,
    _first_dispatch_result,
    _model_handler,
    _parse_unknown_plugin_command,
    _refresh_dispatch_agents,
)
from fast_agent.ui.prompt import parser as prompt_parser

if TYPE_CHECKING:
    from functools import partial

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


def test_unknown_plugin_command_uses_slash_command_line_parser() -> None:
    assert _parse_unknown_plugin_command(UnknownCommand(command="  /Demo   arg one  ")) == (
        "demo",
        "arg one",
    )
    assert _parse_unknown_plugin_command(UnknownCommand(command="plain text")) is None


def test_attachment_token_builds_remote_url_token() -> None:
    token = _attachment_token(
        "https://example.test/path?q=1",
        shell_working_dir=None,
    )

    assert token == "^url:https://example.test/path?q=1"


def test_attachment_token_resolves_relative_file_from_shell_working_dir(tmp_path: Path) -> None:
    attachment = tmp_path / "note.txt"
    attachment.write_text("hello")

    token = _attachment_token("note.txt", shell_working_dir=tmp_path)

    assert token == f"^file:{attachment.as_posix()}"


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


@pytest.mark.asyncio
async def test_prompt_dispatch_propagates_buffer_prefill(monkeypatch) -> None:
    async def fake_run_command_handler(**_kwargs: object) -> CommandOutcome:
        return CommandOutcome(buffer_prefill="loaded prompt")

    monkeypatch.setattr(command_dispatch, "_run_command_handler", fake_run_command_handler)

    result = await _dispatch_prompt_payload(
        LoadPromptCommand(filename="prompt.md", error=None),
        prompt_provider=cast("AgentApp", object()),
        agent="main",
    )

    assert result is not None
    assert result.buffer_prefill == "loaded prompt"


def test_hash_dispatch_returns_buffer_send_details() -> None:
    result = _dispatch_hash_agent_command(
        HashAgentCommand("worker", "summarize", quiet=True),
        available_agents_set={"worker"},
    )

    assert result.hash_send_target == "worker"
    assert result.hash_send_message == "summarize"
    assert result.hash_send_quiet is True


def test_catalog_handler_maps_list_and_action_commands() -> None:
    assert _catalog_handler(ListToolsCommand(), agent="main") is not None
    assert _catalog_handler(CardsCommand(action="list", argument=None), agent="main") is not None
    assert _catalog_handler(ShellCommand("pwd"), agent="main") is None


def test_catalog_handler_preserves_models_command_surface() -> None:
    handler = _catalog_handler(
        ModelsCommand(action="help", argument=None, command_name="models"),
        agent="main",
    )

    assert handler is not None
    partial_handler = cast("partial[object]", handler)
    assert partial_handler.keywords is not None
    assert partial_handler.keywords["command_name"] == "models"


@pytest.mark.asyncio
async def test_history_dispatch_skips_handler_when_target_agent_missing(monkeypatch) -> None:
    async def fail_run_command_handler(**_kwargs: object) -> CommandOutcome:
        raise AssertionError("handler should not run")

    monkeypatch.setattr(command_dispatch, "_run_command_handler", fail_run_command_handler)
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
async def test_history_rewind_dispatch_propagates_buffer_prefill(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_run_command_handler(**kwargs: object) -> CommandOutcome:
        calls.append(str(kwargs["agent"]))
        return CommandOutcome(buffer_prefill="restored prompt")

    monkeypatch.setattr(command_dispatch, "_run_command_handler", fake_run_command_handler)

    result = await _dispatch_history_payload(
        cast("InteractivePrompt", _HistoryOwner()),
        HistoryRewindCommand(turn_index=2, error=None),
        prompt_provider=cast("AgentApp", object()),
        agent="main",
    )

    assert result is not None
    assert calls == ["main"]
    assert result.buffer_prefill == "restored prompt"


@pytest.mark.asyncio
async def test_display_dispatch_runs_mapped_handler(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_run_command_handler(**kwargs: object) -> CommandOutcome:
        calls.append(str(kwargs["agent"]))
        return CommandOutcome()

    monkeypatch.setattr(command_dispatch, "_run_command_handler", fake_run_command_handler)

    result = await _dispatch_display_payload(
        ShowUsageCommand(),
        prompt_provider=cast("AgentApp", object()),
        agent="main",
    )

    assert result is not None
    assert result.handled is True
    assert calls == ["main"]


@pytest.mark.asyncio
async def test_display_dispatch_runs_check_handler(monkeypatch) -> None:
    calls: list[object] = []

    async def fake_run_command_handler(**kwargs: object) -> CommandOutcome:
        calls.append(kwargs["handler"])
        calls.append(kwargs["agent"])
        return CommandOutcome()

    async def fake_check_handler(*_args: object, **_kwargs: object) -> CommandOutcome:
        return CommandOutcome()

    monkeypatch.setattr(command_dispatch, "_run_command_handler", fake_run_command_handler)
    monkeypatch.setattr(command_dispatch.display_handlers, "handle_check", fake_check_handler)

    result = await _dispatch_display_payload(
        CheckCommand(argument="models --for-model gpt-5"),
        prompt_provider=cast("AgentApp", object()),
        agent="main",
    )

    assert result is not None
    assert result.handled is True
    assert calls[1] == "main"
    assert calls[0] is not None


@pytest.mark.asyncio
async def test_display_dispatch_runs_commands_handler(monkeypatch) -> None:
    calls: list[object] = []

    async def fake_run_command_handler(**kwargs: object) -> CommandOutcome:
        calls.append(kwargs["handler"])
        calls.append(kwargs["agent"])
        return CommandOutcome()

    async def fake_commands_handler(*_args: object, **_kwargs: object) -> CommandOutcome:
        return CommandOutcome()

    monkeypatch.setattr(command_dispatch, "_run_command_handler", fake_run_command_handler)
    monkeypatch.setattr(
        command_dispatch.display_handlers,
        "handle_commands",
        fake_commands_handler,
    )

    result = await _dispatch_display_payload(
        CommandsCommand(argument="skills add"),
        prompt_provider=cast("AgentApp", object()),
        agent="main",
    )

    assert result is not None
    assert result.handled is True
    assert calls[1] == "main"
    assert calls[0] is not None


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


def test_history_command_target_agent_handles_targeted_commands() -> None:
    assert command_dispatch._history_command_target_agent(HistoryFixCommand("worker")) == "worker"
    assert (
        command_dispatch._history_command_target_agent(
            ClearCommand(kind="clear_history", agent="worker")
        )
        == "worker"
    )


def test_mcp_server_command_error_requires_server_name() -> None:
    assert command_dispatch._mcp_server_command_error(None, None) == "Server name is required"
    assert command_dispatch._mcp_server_command_error("server", None) is None
    assert command_dispatch._mcp_server_command_error("server", "bad") == "bad"


@pytest.mark.asyncio
async def test_mcp_dispatch_runs_handler_for_list_command(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_run_command_handler(**kwargs: object) -> CommandOutcome:
        calls.append(str(kwargs["agent"]))
        return CommandOutcome()

    monkeypatch.setattr(command_dispatch, "_run_command_handler", fake_run_command_handler)

    result = await _dispatch_mcp_payload(
        McpListCommand(),
        prompt_provider=cast("AgentApp", object()),
        agent="main",
    )

    assert result is not None
    assert result.handled is True
    assert calls == ["main"]


@pytest.mark.asyncio
async def test_mcp_dispatch_skips_handler_when_server_name_missing(monkeypatch) -> None:
    async def fail_run_command_handler(**_kwargs: object) -> CommandOutcome:
        raise AssertionError("handler should not run")

    monkeypatch.setattr(command_dispatch, "_run_command_handler", fail_run_command_handler)

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


def test_mcp_handler_returns_none_for_session_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    printed: list[object] = []
    monkeypatch.setattr(command_dispatch, "rich_print", printed.append)

    handler = command_dispatch._mcp_handler(
        McpSessionCommand(
            action="list",
            server_identity=None,
            session_id=None,
            title=None,
            clear_all=False,
            error="bad [session] command",
        ),
        prompt_provider=cast("AgentApp", object()),
        agent="main",
    )

    assert handler is None
    assert [getattr(item, "plain", item) for item in printed] == ["bad [session] command"]


def test_mcp_handler_maps_server_commands() -> None:
    handler = command_dispatch._mcp_handler(
        McpReconnectCommand(server_name="docs", error=None),
        prompt_provider=cast("AgentApp", object()),
        agent="main",
    )

    assert handler is not None


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


def test_model_handler_maps_value_commands_but_not_switch() -> None:
    assert _model_handler(ModelWebFetchCommand("on"), agent="main") is not None
    assert _model_handler(ModelSwitchCommand("openai.gpt-5"), agent="main") is None


@pytest.mark.asyncio
async def test_session_dispatch_runs_generic_handler(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_run_command_handler(**kwargs: object) -> CommandOutcome:
        calls.append(str(kwargs["agent"]))
        return CommandOutcome()

    monkeypatch.setattr(command_dispatch, "_run_command_handler", fake_run_command_handler)

    result = await _dispatch_session_payload(
        TitleSessionCommand("new title"),
        prompt_provider=cast("AgentApp", object()),
        agent="main",
    )

    assert result is not None
    assert result.handled is True
    assert calls == ["main"]


@pytest.mark.asyncio
async def test_session_resume_dispatch_propagates_switch_agent(monkeypatch) -> None:
    async def fake_run_command_handler(**_kwargs: object) -> CommandOutcome:
        return CommandOutcome(switch_agent="worker")

    monkeypatch.setattr(command_dispatch, "_run_command_handler", fake_run_command_handler)

    result = await _dispatch_session_payload(
        ResumeSessionCommand("session-1"),
        prompt_provider=cast("AgentApp", object()),
        agent="main",
    )

    assert result is not None
    assert result.next_agent == "worker"


def test_session_handler_ignores_non_session_payload() -> None:
    assert command_dispatch._session_handler(ShellCommand("pwd"), agent="main") is None


def test_parse_unknown_plugin_command_requires_slash_command() -> None:
    assert command_dispatch._parse_unknown_plugin_command(
        command_dispatch.UnknownCommand("draft")
    ) is None


def test_parse_unknown_plugin_command_splits_arguments() -> None:
    parsed = command_dispatch._parse_unknown_plugin_command(
        command_dispatch.UnknownCommand("/Draft  a message")
    )

    assert parsed == ("draft", "a message")


def test_plugin_action_outcome_prefers_markdown_message() -> None:
    outcome = command_dispatch._plugin_action_outcome(
        PluginCommandActionResult(
            message="plain",
            markdown="**markdown**",
            buffer_prefill="draft",
            switch_agent="worker",
            refresh_agents=True,
        )
    )

    assert outcome.buffer_prefill == "draft"
    assert outcome.switch_agent == "worker"
    assert outcome.requires_refresh is True
    assert outcome.messages[0].text == "**markdown**"
    assert outcome.messages[0].render_markdown is True


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


def test_refresh_dispatch_agents_records_available_agents() -> None:
    result = command_dispatch.DispatchResult(handled=True)
    owner = _RefreshOwner()

    agents, agent_set = _refresh_dispatch_agents(
        result,
        owner=cast("InteractivePrompt", owner),
        prompt_provider=cast("AgentApp", _RefreshProvider()),
        merge_pinned_agents=lambda names: [*names, "pinned"],
    )

    assert agents == ["main", "worker", "pinned"]
    assert agent_set == {"main", "worker", "pinned"}
    assert result.available_agents == agents
    assert result.available_agents_set == agent_set
    assert owner.agent_types == {"main": "basic"}


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


@pytest.mark.asyncio
async def test_first_dispatch_result_returns_first_handled_result() -> None:
    calls: list[str] = []

    async def miss() -> command_dispatch.DispatchResult | None:
        calls.append("miss")
        return None

    async def hit() -> command_dispatch.DispatchResult:
        calls.append("hit")
        return command_dispatch.DispatchResult(handled=True, shell_execute_cmd="pwd")

    result = await _first_dispatch_result(
        (
            _DispatchStep(name="first", run=miss),
            _DispatchStep(name="second", run=hit),
        )
    )

    assert result is not None
    assert result.shell_execute_cmd == "pwd"
    assert calls == ["miss", "hit"]
