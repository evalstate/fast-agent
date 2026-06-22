from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import typer
from mcp import CallToolRequest
from mcp.types import CallToolRequestParams, TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.cli.runtime.agent_setup import (
    _apply_fast_args,
    _apply_shell_cwd_policy_preflight,
    _build_fan_out_result_paths,
    _build_result_file_with_suffix,
    _cli_attachment_token,
    _export_result_histories,
    _find_last_assistant_text,
    _resume_session_if_requested,
    _run_cli_flow,
    _select_loaded_card_agent,
)
from fast_agent.cli.runtime.harness_startup import (
    run_cli_flow,
    should_use_harness_startup,
)
from fast_agent.cli.runtime.run_request import AgentRunRequest
from fast_agent.cli.runtime.runner import _should_convert_keyboard_interrupt_to_task_cancel
from fast_agent.config import Settings, ShellSettings
from fast_agent.core.exceptions import AgentConfigError
from fast_agent.core.harness_app import DefaultHarnessApp
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.prompt_serialization import load_messages
from fast_agent.session import ResumeSessionAgentsResult
from fast_agent.session.hydrator import SessionHydrationWarning
from fast_agent.types.llm_stop_reason import LlmStopReason


class _DummyAgent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.message_history: list[object] = []


class _NonPersistentMessageAgent(_DummyAgent):
    def __init__(self, name: str, reply_text: str) -> None:
        super().__init__(name)
        self.reply_text = reply_text
        self.generated_messages: list[object] = []

    async def generate(self, messages: object) -> PromptMessageExtended:
        self.generated_messages.append(messages)
        return PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text=self.reply_text)],
        )


class _StructuredMessageAgent(_DummyAgent):
    def __init__(self, name: str, payload: object | None, reply_text: str) -> None:
        super().__init__(name)
        self.payload = payload
        self.reply_text = reply_text
        self.generated_messages: list[object] = []
        self.schemas: list[dict[str, Any]] = []

    async def structured_schema(
        self,
        messages: object,
        schema: dict[str, Any],
    ) -> tuple[object | None, PromptMessageExtended]:
        self.generated_messages.append(messages)
        self.schemas.append(schema)
        return self.payload, PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text=self.reply_text)],
        )


class _DummyAgentApp:
    def __init__(self, agent_names: list[str], *, default_agent: str | None = None) -> None:
        self._agents = {name: _DummyAgent(name) for name in agent_names}
        self._default_agent = default_agent or agent_names[0]
        self._session_restore_result = None

    def _agent(self, agent_name: str | None):
        if agent_name is None:
            return self._agents[self._default_agent]
        return self._agents[agent_name]

    def get_agent(self, agent_name: str):
        return self._agents.get(agent_name)

    def resolve_target_agent_name(self, agent_name: str | None = None) -> str | None:
        return self._default_agent if agent_name is None else agent_name

    def resolve_agent(self, agent_name: str | None = None):
        return self._agent(self.resolve_target_agent_name(agent_name))

    def registered_agents(self):
        return self._agents

    def latest_session_restore_result(self):
        return self._session_restore_result

    async def interactive(
        self,
        agent_name: str | None = None,
        session_manager: object | None = None,
        harness_session: object | None = None,
    ) -> None:
        del agent_name, session_manager, harness_session

    async def send(self, message: str, agent_name: str | None = None) -> str:
        del agent_name
        return message


class _AsyncContext:
    def __init__(self, value: object) -> None:
        self.value = value

    async def __aenter__(self) -> object:
        return self.value

    async def __aexit__(self, *args: object) -> None:
        del args


class _FailingAsyncContext(_AsyncContext):
    def __init__(self, exc: Exception) -> None:
        super().__init__(object())
        self.exc = exc

    async def __aenter__(self) -> object:
        raise self.exc


class _DummyHarness:
    def __init__(self, agent_app: _DummyAgentApp, session_manager: object) -> None:
        self.agent_app = agent_app
        self.session_manager = session_manager
        self.generated_messages: list[tuple[object, str | None]] = []

    async def session(self, session_id: str, *, agent_name: str | None = None):
        del session_id, agent_name
        return self

    def app(self):
        return DefaultHarnessApp(cast("Any", self))

    async def generate(
        self,
        messages: object,
        *,
        agent_name: str | None = None,
        request_params: object | None = None,
    ) -> PromptMessageExtended:
        del request_params
        self.generated_messages.append((messages, agent_name))
        return await self.agent_app._agent(agent_name).generate(messages)

    async def structured_schema(
        self,
        messages: object,
        schema: dict[str, Any],
        *,
        agent_name: str | None = None,
        request_params: object | None = None,
    ) -> tuple[object | None, PromptMessageExtended]:
        del request_params
        self.generated_messages.append((messages, agent_name))
        return await self.agent_app._agent(agent_name).structured_schema(messages, schema)


class _DummyFastRuntime:
    def __init__(self) -> None:
        self.direct_app = _DummyAgentApp(["agent"])
        self.harness_app = _DummyAgentApp(["agent"])
        self.session_manager = object()
        self.harness_session = _DummyHarness(self.harness_app, self.session_manager)
        self.run_calls = 0
        self.harness_calls = 0

    def run(self) -> _AsyncContext:
        self.run_calls += 1
        return _AsyncContext(self.direct_app)

    def harness(self) -> _AsyncContext:
        self.harness_calls += 1
        return _AsyncContext(self.harness_session)


class _FailingHarnessRuntime(_DummyFastRuntime):
    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self.exc = exc
        self.handled_errors: list[Exception] = []

    def harness(self) -> _FailingAsyncContext:
        self.harness_calls += 1
        return _FailingAsyncContext(self.exc)

    def _handle_error(self, exc: Exception, error_type: str | None = None) -> None:
        del error_type
        self.handled_errors.append(exc)


def _make_request(
    *,
    result_file: str | None,
    target_agent_name: str | None = None,
    message: str | None = "hello",
    prompt_file: str | None = None,
    attachments: list[str] | None = None,
    json_schema: str | None = None,
) -> AgentRunRequest:
    return AgentRunRequest(
        name="test",
        instruction="instruction",
        config_path=None,
        server_list=None,
        agent_cards=None,
        card_tools=None,
        model=None,
        message=message,
        prompt_file=prompt_file,
        json_schema=json_schema,
        result_file=result_file,
        resume=None,
        url_servers=None,
        stdio_servers=None,
        agent_name="agent",
        target_agent_name=target_agent_name,
        skills_directory=None,
        environment_dir=None,
        noenv=False,
        force_smart=False,
        shell_runtime=False,
        no_shell=False,
        mode="interactive",
        transport="http",
        host="127.0.0.1",
        port=8000,
        tool_description=None,
        tool_name_template=None,
        instance_scope="shared",
        permissions_enabled=True,
        reload=False,
        watch=False,
        attachments=attachments,
    )


def test_build_result_file_with_suffix_without_extension() -> None:
    assert _build_result_file_with_suffix(Path("foo"), "haiku35") == Path("foo-haiku35")


def test_select_loaded_card_agent_targets_single_runnable_card() -> None:
    request = _make_request(result_file=None, message=None)
    fast = SimpleNamespace(agents={"news": {"tool_only": False}})

    selected = _select_loaded_card_agent(fast, request, ["news"])

    assert selected == "news"
    assert request.target_agent_name == "news"


def test_select_loaded_card_agent_leaves_ambiguous_cards_unselected() -> None:
    request = _make_request(result_file=None, message=None)
    fast = SimpleNamespace(
        agents={
            "news": {"tool_only": False},
            "summary": {"tool_only": False},
        }
    )

    selected = _select_loaded_card_agent(fast, request, ["news", "summary"])

    assert selected is None
    assert request.target_agent_name is None


def test_should_convert_keyboard_interrupt_to_task_cancel_only_for_interactive_repl() -> None:
    interactive_request = _make_request(result_file=None, message=None)
    assert _should_convert_keyboard_interrupt_to_task_cancel(interactive_request) is True

    one_shot_request = _make_request(result_file=None, message="hello")
    assert _should_convert_keyboard_interrupt_to_task_cancel(one_shot_request) is False

    prompt_file_request = _make_request(
        result_file=None,
        message=None,
        prompt_file="prompt.txt",
    )
    assert _should_convert_keyboard_interrupt_to_task_cancel(prompt_file_request) is False


def test_apply_fast_args_threads_resume_into_runtime_settings() -> None:
    request = _make_request(result_file=None)
    request.resume = "session-123"
    fast = SimpleNamespace(args=SimpleNamespace())

    _apply_fast_args(fast, request)

    assert fast.args.resume_requested is True
    assert fast.args.resume_session_id == "session-123"


def test_apply_fast_args_rejects_noenv_resume_before_enabling_restore() -> None:
    request = _make_request(result_file=None)
    request.noenv = True
    request.resume = "session-123"
    fast = SimpleNamespace(args=SimpleNamespace())

    with pytest.raises(typer.Exit):
        _apply_fast_args(fast, request)

    assert "resume_requested" not in vars(fast.args)


@pytest.mark.asyncio
async def test_run_cli_flow_uses_harness_for_local_repl() -> None:
    request = _make_request(result_file=None, message=None)
    fast = _DummyFastRuntime()
    calls: list[tuple[object, object | None, object | None]] = []

    async def flow(
        agent_app: object,
        request: AgentRunRequest,
        *,
        session_manager: object | None = None,
        harness_session: object | None = None,
    ) -> None:
        del request
        calls.append((agent_app, session_manager, harness_session))

    assert should_use_harness_startup(request)

    await run_cli_flow(cast("Any", fast), request, flow=flow)

    assert fast.harness_calls == 1
    assert fast.run_calls == 0
    assert calls == [(fast.harness_app, fast.session_manager, fast.harness_session)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("resume", "latest"),
        ("noenv", True),
    ],
)
async def test_run_cli_flow_uses_direct_run_for_resume_and_noenv(
    field: str,
    value: object,
) -> None:
    request = _make_request(result_file=None, message=None)
    setattr(request, field, value)
    fast = _DummyFastRuntime()
    calls: list[tuple[object, object | None, object | None]] = []

    async def flow(
        agent_app: object,
        request: AgentRunRequest,
        *,
        session_manager: object | None = None,
        harness_session: object | None = None,
    ) -> None:
        del request
        calls.append((agent_app, session_manager, harness_session))

    assert not should_use_harness_startup(request)

    await run_cli_flow(cast("Any", fast), request, flow=flow)

    assert fast.harness_calls == 0
    assert fast.run_calls == 1
    assert calls == [(fast.direct_app, None, None)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value", "execution_mode"),
    [
        ("message", "hello", "one_shot_message"),
        ("prompt_file", "prompt.json", "one_shot_prompt_file"),
    ],
)
async def test_run_cli_flow_uses_harness_for_one_shot_modes(
    field: str,
    value: object,
    execution_mode: str,
) -> None:
    request = _make_request(result_file=None, message=None)
    setattr(request, field, value)
    request.execution_mode = cast("Any", execution_mode)
    fast = _DummyFastRuntime()
    calls: list[tuple[object, object | None, object | None]] = []

    async def flow(
        agent_app: object,
        request: AgentRunRequest,
        *,
        session_manager: object | None = None,
        harness_session: object | None = None,
    ) -> None:
        del request
        calls.append((agent_app, session_manager, harness_session))

    assert should_use_harness_startup(request)

    await run_cli_flow(cast("Any", fast), request, flow=flow)

    assert fast.harness_calls == 1
    assert fast.run_calls == 0
    assert calls == [(fast.harness_app, fast.session_manager, fast.harness_session)]


@pytest.mark.asyncio
async def test_run_cli_flow_handles_harness_startup_errors_like_cli_run() -> None:
    request = _make_request(result_file=None, message=None)
    error = AgentConfigError("bad agent")
    fast = _FailingHarnessRuntime(error)

    async def flow(
        agent_app: object,
        request: AgentRunRequest,
        *,
        session_manager: object | None = None,
        harness_session: object | None = None,
    ) -> None:
        del agent_app, request, session_manager, harness_session

    with pytest.raises(SystemExit) as exc_info:
        await run_cli_flow(cast("Any", fast), request, flow=flow)

    assert exc_info.value.code == 1
    assert fast.handled_errors == [error]


def test_build_fan_out_result_paths_disambiguates_collisions() -> None:
    exports = _build_fan_out_result_paths(
        "foo.json",
        ["alpha/beta", "alpha beta", "alpha\\beta"],
    )
    assert [path.name for _, path in exports] == [
        "foo-alpha_beta.json",
        "foo-alpha_beta-2.json",
        "foo-alpha_beta-3.json",
    ]


def test_find_last_assistant_text_prefers_latest_assistant_message() -> None:
    history = [
        PromptMessageExtended(role="user", content=[TextContent(type="text", text="hello")]),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="first")],
        ),
        PromptMessageExtended(role="user", content=[TextContent(type="text", text="again")]),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="second")],
        ),
    ]

    assert _find_last_assistant_text(history) == "second"


def test_find_last_assistant_text_returns_none_without_assistant_messages() -> None:
    history = [PromptMessageExtended(role="user", content=[TextContent(type="text", text="hello")])]

    assert _find_last_assistant_text(history) is None


def test_find_last_assistant_text_falls_back_to_pending_tool_summary() -> None:
    history = [
        PromptMessageExtended(role="user", content=[TextContent(type="text", text="hello")]),
        PromptMessageExtended(
            role="assistant",
            tool_calls={
                "call-1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="read_text_file", arguments={}),
                )
            },
            stop_reason=LlmStopReason.TOOL_USE,
        ),
    ]

    assert _find_last_assistant_text(history) == "Pending tool call: read_text_file"


def test_find_last_assistant_text_counts_unnamed_pending_tools() -> None:
    history = [
        PromptMessageExtended(
            role="assistant",
            tool_calls={
                "call-1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="", arguments={}),
                ),
                "call-2": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="", arguments={}),
                ),
            },
            stop_reason=LlmStopReason.TOOL_USE,
        ),
    ]

    assert _find_last_assistant_text(history) == "Pending 2 tool calls"


def test_find_last_assistant_text_prefers_text_over_pending_tool_summary() -> None:
    history = [
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="partial answer")],
            tool_calls={
                "call-1": CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(name="read_text_file", arguments={}),
                )
            },
            stop_reason=LlmStopReason.TOOL_USE,
        )
    ]

    assert _find_last_assistant_text(history) == "partial answer"


@pytest.mark.asyncio
async def test_run_cli_flow_exports_transient_turn_when_history_disabled(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = _NonPersistentMessageAgent("agent", "done")
    app = _DummyAgentApp(["agent"])
    app._agents["agent"] = agent
    output = tmp_path / "out.json"

    await _run_cli_flow(app, _make_request(result_file=str(output), message="hello"))

    assert agent.message_history == []
    exported = load_messages(str(output))
    assert [message.role for message in exported] == ["user", "assistant"]
    assert exported[0].first_text() == "hello"
    assert exported[1].last_text() == "done"
    captured = capsys.readouterr()
    assert captured.out.strip() == "done"


@pytest.mark.asyncio
async def test_run_cli_flow_one_shot_uses_harness_session_when_available(
    capsys: pytest.CaptureFixture[str],
) -> None:
    agent = _NonPersistentMessageAgent("agent", "done")
    app = _DummyAgentApp(["agent"])
    app._agents["agent"] = agent
    harness_session = _DummyHarness(app, object())

    await _run_cli_flow(
        app,
        _make_request(result_file=None, message="hello"),
        harness_session=cast("Any", harness_session),
    )

    assert harness_session.generated_messages == [("hello", "agent")]
    captured = capsys.readouterr()
    assert captured.out.strip() == "done"


@pytest.mark.asyncio
async def test_run_cli_flow_prompt_file_is_one_shot_and_exports_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _TrackingAgentApp(_DummyAgentApp):
        def __init__(self) -> None:
            super().__init__(["agent"])
            self.interactive_calls = 0

        async def interactive(
            self,
            agent_name: str | None = None,
            session_manager: object | None = None,
            harness_session: object | None = None,
        ) -> None:
            del agent_name, session_manager, harness_session
            self.interactive_calls += 1

    prompt = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="hello from prompt")],
        )
    ]
    prompt_file = tmp_path / "prompt.json"
    prompt_file.write_text("[]", encoding="utf-8")
    output = tmp_path / "out.json"

    agent = _NonPersistentMessageAgent("agent", "done")
    app = _TrackingAgentApp()
    app._agents["agent"] = agent

    monkeypatch.setattr(
        "fast_agent.mcp.prompts.prompt_load.load_prompt",
        lambda _path: prompt,
    )

    await _run_cli_flow(
        app,
        _make_request(
            result_file=str(output),
            message=None,
            prompt_file=str(prompt_file),
        ),
    )

    assert app.interactive_calls == 0
    exported = load_messages(str(output))
    assert [message.role for message in exported] == ["user", "assistant"]
    assert exported[0].first_text() == "hello from prompt"
    assert exported[1].last_text() == "done"
    captured = capsys.readouterr()
    assert captured.out.strip() == "done"


@pytest.mark.asyncio
async def test_run_cli_flow_message_attaches_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    attachment = tmp_path / "report.txt"
    attachment.write_text("attached text", encoding="utf-8")
    agent = _NonPersistentMessageAgent("agent", "done")
    app = _DummyAgentApp(["agent"])
    app._agents["agent"] = agent

    await _run_cli_flow(
        app,
        _make_request(
            result_file=None,
            message="summarize",
            attachments=[attachment.as_posix()],
        ),
    )

    sent = agent.generated_messages[0]
    assert isinstance(sent, PromptMessageExtended)
    assert sent.first_text() == "summarize"
    assert len(sent.content) == 2
    assert capsys.readouterr().out.strip() == "done"


def test_cli_attachment_token_normalizes_remote_scheme_case() -> None:
    assert _cli_attachment_token("HTTPS://example.com/report.pdf").startswith(
        "^url:HTTPS://example.com/report.pdf"
    )


@pytest.mark.asyncio
async def test_run_cli_flow_prompt_file_attaches_to_last_user_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    attachment = tmp_path / "report.txt"
    attachment.write_text("attached text", encoding="utf-8")
    prompt = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="first")],
        ),
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="middle")],
        ),
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="last")],
        ),
    ]
    prompt_file = tmp_path / "prompt.json"
    prompt_file.write_text("[]", encoding="utf-8")
    agent = _NonPersistentMessageAgent("agent", "done")
    app = _DummyAgentApp(["agent"])
    app._agents["agent"] = agent

    monkeypatch.setattr(
        "fast_agent.mcp.prompts.prompt_load.load_prompt",
        lambda _path: prompt,
    )

    await _run_cli_flow(
        app,
        _make_request(
            result_file=None,
            message=None,
            prompt_file=str(prompt_file),
            attachments=[attachment.as_posix()],
        ),
    )

    sent = agent.generated_messages[0]
    assert isinstance(sent, list)
    sent_messages = cast("list[PromptMessageExtended]", sent)
    assert [len(message.content) for message in sent_messages] == [1, 1, 2]
    assert sent_messages[2].first_text() == "last"
    assert len(prompt[2].content) == 1
    assert capsys.readouterr().out.strip() == "done"


@pytest.mark.asyncio
async def test_run_cli_flow_json_schema_message_emits_only_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    schema_path = tmp_path / "schema.json"
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    agent = _StructuredMessageAgent("agent", {"answer": "done"}, '{"answer":"done"}')
    app = _DummyAgentApp(["agent"])
    app._agents["agent"] = agent

    await _run_cli_flow(
        app,
        _make_request(
            result_file=None,
            message="hello",
            json_schema=str(schema_path),
        ),
    )

    captured = capsys.readouterr()
    assert captured.out == '{"answer": "done"}'
    assert captured.err == ""
    assert agent.schemas == [schema]


@pytest.mark.asyncio
async def test_run_cli_flow_json_schema_prompt_file_emits_only_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    schema_path = tmp_path / "schema.json"
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    schema_path.write_text(json.dumps(schema), encoding="utf-8")

    prompt = [
        PromptMessageExtended(
            role="user",
            content=[TextContent(type="text", text="hello from prompt")],
        )
    ]
    prompt_file = tmp_path / "prompt.json"
    prompt_file.write_text("[]", encoding="utf-8")

    agent = _StructuredMessageAgent("agent", {"answer": "done"}, '{"answer":"done"}')
    app = _DummyAgentApp(["agent"])
    app._agents["agent"] = agent

    monkeypatch.setattr(
        "fast_agent.mcp.prompts.prompt_load.load_prompt",
        lambda _path: prompt,
    )

    await _run_cli_flow(
        app,
        _make_request(
            result_file=None,
            message=None,
            prompt_file=str(prompt_file),
            json_schema=str(schema_path),
        ),
    )

    captured = capsys.readouterr()
    assert captured.out == '{"answer": "done"}'
    assert captured.err == ""
    assert agent.generated_messages == [prompt]


@pytest.mark.asyncio
async def test_run_cli_flow_json_schema_invalid_output_exits_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    schema_path = tmp_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            }
        ),
        encoding="utf-8",
    )

    agent = _StructuredMessageAgent("agent", None, "not-json")
    app = _DummyAgentApp(["agent"])
    app._agents["agent"] = agent

    with pytest.raises(typer.Exit) as exc_info:
        await _run_cli_flow(
            app,
            _make_request(
                result_file=None,
                message="hello",
                json_schema=str(schema_path),
            ),
        )

    assert exc_info.value.exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "valid JSON matching the structured output schema" in captured.err


@pytest.mark.asyncio
async def test_export_result_histories_single_agent_exact_filename(tmp_path: Path) -> None:
    app = _DummyAgentApp(["agent"])
    output = tmp_path / "out.json"

    await _export_result_histories(app, _make_request(result_file=str(output)))

    assert output.exists()


@pytest.mark.asyncio
async def test_export_result_histories_multi_model_writes_suffixed_files(tmp_path: Path) -> None:
    app = _DummyAgentApp(["glm4.7", "haiku35"], default_agent="glm4.7")
    output = tmp_path / "out.json"

    await _export_result_histories(
        app,
        _make_request(result_file=str(output)),
        fan_out_agent_names=["glm4.7", "haiku35"],
    )

    assert (tmp_path / "out-glm4.7.json").exists()
    assert (tmp_path / "out-haiku35.json").exists()


@pytest.mark.asyncio
async def test_export_result_histories_multi_model_with_target_exports_exact_filename(
    tmp_path: Path,
) -> None:
    app = _DummyAgentApp(["glm4.7", "haiku35"], default_agent="glm4.7")
    output = tmp_path / "out.json"

    await _export_result_histories(
        app,
        _make_request(result_file=str(output), target_agent_name="haiku35"),
        fan_out_agent_names=["glm4.7", "haiku35"],
    )

    assert output.exists()
    assert not (tmp_path / "out-glm4.7.json").exists()
    assert not (tmp_path / "out-haiku35.json").exists()


@pytest.mark.asyncio
async def test_export_result_histories_exits_nonzero_on_write_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    app = _DummyAgentApp(["agent"])
    not_a_dir = tmp_path / "not-a-dir"
    not_a_dir.write_text("content", encoding="utf-8")
    output = not_a_dir / "out.json"

    with pytest.raises(typer.Exit) as exc_info:
        await _export_result_histories(app, _make_request(result_file=str(output)))

    assert exc_info.value.exit_code == 1
    captured = capsys.readouterr()
    assert "Error exporting result file" in captured.err


@pytest.mark.asyncio
async def test_resume_session_interactive_queues_markdown_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_request(result_file=None, message=None, prompt_file=None)
    request.resume = "latest"

    assistant_message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="## Welcome back\n\n- item")],
    )

    alpha = _DummyAgent("alpha")
    alpha.message_history = [assistant_message]
    beta = _DummyAgent("beta")

    app = _DummyAgentApp(["alpha", "beta"], default_agent="beta")
    app._agents["alpha"] = alpha
    app._agents["beta"] = beta

    session = SimpleNamespace(
        info=SimpleNamespace(
            name="session-1",
            last_activity=datetime(2026, 2, 26, 12, 0, 0),
        )
    )

    app._session_restore_result = ResumeSessionAgentsResult(
        session=cast("Any", session),
        loaded={"alpha": Path("history_alpha.json")},
        missing_agents=[],
    )

    markdown_notices: list[tuple[str, dict[str, str | None]]] = []
    plain_notices: list[str] = []

    def _capture_markdown_notice(text: str, **kwargs: str | None) -> None:
        markdown_notices.append((text, kwargs))

    monkeypatch.setattr("fast_agent.ui.enhanced_prompt.queue_startup_notice", plain_notices.append)
    monkeypatch.setattr(
        "fast_agent.ui.enhanced_prompt.queue_startup_markdown_notice",
        _capture_markdown_notice,
    )

    await _resume_session_if_requested(app, request)

    assert any("Resumed session" in notice for notice in plain_notices)
    assert markdown_notices
    assert markdown_notices[0][0] == "## Welcome back\n\n- item"
    assert markdown_notices[0][1]["title"] == "Last assistant message"


@pytest.mark.asyncio
async def test_resume_session_interactive_queues_git_warning_after_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_request(result_file=None, message=None, prompt_file=None)
    request.resume = "latest"

    agent = _DummyAgent("agent")
    agent.message_history = [
        PromptMessageExtended(
            role="assistant",
            content=[TextContent(type="text", text="last response")],
        )
    ]
    app = _DummyAgentApp(["agent"], default_agent="agent")
    app._agents["agent"] = agent

    session = SimpleNamespace(
        info=SimpleNamespace(
            name="session-1",
            last_activity=datetime(2026, 2, 26, 12, 0, 0),
        )
    )
    app._session_restore_result = ResumeSessionAgentsResult(
        session=cast("Any", session),
        loaded={"agent": Path("history_agent.json")},
        missing_agents=[],
        warnings=[
            SessionHydrationWarning(
                code="git-state-changed",
                message="Git state changed since session save: commit 746b5a9 -> af77669.",
            )
        ],
    )

    events: list[str] = []

    monkeypatch.setattr(
        "fast_agent.ui.enhanced_prompt.queue_startup_notice",
        lambda notice: events.append(str(notice)),
    )
    monkeypatch.setattr(
        "fast_agent.ui.enhanced_prompt.queue_startup_markdown_notice",
        lambda text, **kwargs: events.append(f"preview:{text}"),
    )

    await _resume_session_if_requested(app, request)

    assert events[-2:] == [
        "preview:last response",
        "[yellow]Git state changed since session save: commit 746b5a9 -> af77669.[/yellow]",
    ]


@pytest.mark.asyncio
async def test_resume_session_preview_fallback_preserves_loaded_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_request(result_file=None, message=None, prompt_file=None)
    request.resume = "latest"

    beta_message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="beta preview")],
    )
    alpha_message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="alpha preview")],
    )
    beta = _DummyAgent("beta")
    beta.message_history = [beta_message]
    alpha = _DummyAgent("alpha")
    alpha.message_history = [alpha_message]

    app = _DummyAgentApp(["alpha", "beta", "default"], default_agent="default")
    app._agents["alpha"] = alpha
    app._agents["beta"] = beta

    session = SimpleNamespace(
        info=SimpleNamespace(
            name="session-ordered",
            last_activity=datetime(2026, 2, 26, 12, 0, 0),
        )
    )

    app._session_restore_result = ResumeSessionAgentsResult(
        session=cast("Any", session),
        loaded={
            "beta": Path("history_beta.json"),
            "alpha": Path("history_alpha.json"),
        },
        missing_agents=[],
    )
    markdown_notices: list[tuple[str, dict[str, str | None]]] = []

    def _capture_markdown_notice(text: str, **kwargs: str | None) -> None:
        markdown_notices.append((text, kwargs))

    monkeypatch.setattr("fast_agent.ui.enhanced_prompt.queue_startup_notice", lambda *_args: None)
    monkeypatch.setattr(
        "fast_agent.ui.enhanced_prompt.queue_startup_markdown_notice",
        _capture_markdown_notice,
    )

    await _resume_session_if_requested(app, request)

    assert markdown_notices
    assert markdown_notices[0][0] == "beta preview"


@pytest.mark.asyncio
async def test_resume_session_interactive_handles_usage_notices_from_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_request(result_file=None, message=None, prompt_file=None)
    request.resume = "latest"

    assistant_message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="done")],
    )
    agent = _DummyAgent("agent")
    agent.message_history = [assistant_message]
    app = _DummyAgentApp(["agent"], default_agent="agent")
    app._agents["agent"] = agent

    session = SimpleNamespace(
        info=SimpleNamespace(
            name="session-2",
            last_activity=datetime(2026, 2, 26, 12, 0, 0),
        )
    )

    app._session_restore_result = ResumeSessionAgentsResult(
        session=cast("Any", session),
        loaded={"agent": Path("history_agent.json")},
        missing_agents=[],
        usage_notices=["[dim]Usage restored[/dim]"],
    )

    plain_notices: list[str] = []

    monkeypatch.setattr("fast_agent.ui.enhanced_prompt.queue_startup_notice", plain_notices.append)
    monkeypatch.setattr(
        "fast_agent.ui.enhanced_prompt.queue_startup_markdown_notice",
        lambda *args, **kwargs: None,
    )

    await _resume_session_if_requested(app, request)

    assert any("Usage restored" in notice for notice in plain_notices)


@pytest.mark.asyncio
async def test_resume_session_applies_hydrated_active_agent_to_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_request(result_file=None, message=None, prompt_file=None)
    request.resume = "latest"

    alpha_message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="alpha preview")],
    )
    beta_message = PromptMessageExtended(
        role="assistant",
        content=[TextContent(type="text", text="beta preview")],
    )
    alpha = _DummyAgent("alpha")
    alpha.message_history = [alpha_message]
    beta = _DummyAgent("beta")
    beta.message_history = [beta_message]

    app = _DummyAgentApp(["alpha", "beta"], default_agent="beta")
    app._agents["alpha"] = alpha
    app._agents["beta"] = beta

    session = SimpleNamespace(
        info=SimpleNamespace(
            name="session-2b",
            last_activity=datetime(2026, 2, 26, 12, 0, 0),
        )
    )

    app._session_restore_result = ResumeSessionAgentsResult(
        session=cast("Any", session),
        loaded={
            "alpha": Path("history_alpha.json"),
            "beta": Path("history_beta.json"),
        },
        missing_agents=[],
        active_agent="alpha",
    )

    markdown_notices: list[tuple[str, dict[str, str | None]]] = []

    def _capture_markdown_notice(text: str, **kwargs: str | None) -> None:
        markdown_notices.append((text, kwargs))

    monkeypatch.setattr("fast_agent.ui.enhanced_prompt.queue_startup_notice", lambda *_args: None)
    monkeypatch.setattr(
        "fast_agent.ui.enhanced_prompt.queue_startup_markdown_notice",
        _capture_markdown_notice,
    )

    await _resume_session_if_requested(app, request)

    assert request.target_agent_name == "alpha"
    assert markdown_notices
    assert markdown_notices[0][0] == "alpha preview"


@pytest.mark.asyncio
async def test_resume_session_prefers_explicit_target_agent_for_fallback_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _make_request(
        result_file=None,
        target_agent_name="beta",
        message=None,
        prompt_file=None,
    )
    request.resume = "latest"

    app = _DummyAgentApp(["alpha", "beta"], default_agent="alpha")
    session = SimpleNamespace(
        info=SimpleNamespace(
            name="session-3",
            last_activity=datetime(2026, 2, 26, 12, 0, 0),
        )
    )
    app._session_restore_result = ResumeSessionAgentsResult(
        session=cast("Any", session),
        loaded={"beta": Path("history_beta.json")},
        missing_agents=[],
    )
    monkeypatch.setattr("fast_agent.ui.enhanced_prompt.queue_startup_notice", lambda *_args: None)
    monkeypatch.setattr(
        "fast_agent.ui.enhanced_prompt.queue_startup_markdown_notice",
        lambda *args, **kwargs: None,
    )

    await _resume_session_if_requested(app, request)

    assert request.target_agent_name == "beta"


@pytest.mark.asyncio
async def test_run_cli_flow_retries_interactive_after_keyboard_interrupt(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _InterruptingAgentApp(_DummyAgentApp):
        def __init__(self) -> None:
            super().__init__(["agent"])
            self.interactive_calls = 0

        async def interactive(
            self,
            agent_name: str | None = None,
            session_manager: object | None = None,
            harness_session: object | None = None,
        ) -> None:
            del agent_name, session_manager, harness_session
            self.interactive_calls += 1
            if self.interactive_calls == 1:
                raise KeyboardInterrupt()

    app = _InterruptingAgentApp()
    request = _make_request(result_file=None, message=None)

    await _run_cli_flow(app, request)

    captured = capsys.readouterr()
    assert app.interactive_calls == 2
    assert "Interrupted operation; returning to fast-agent prompt." in captured.err


@pytest.mark.asyncio
async def test_run_cli_flow_exits_after_double_keyboard_interrupt(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _InterruptingAgentApp(_DummyAgentApp):
        def __init__(self) -> None:
            super().__init__(["agent"])
            self.interactive_calls = 0

        async def interactive(
            self,
            agent_name: str | None = None,
            session_manager: object | None = None,
            harness_session: object | None = None,
        ) -> None:
            del agent_name, session_manager, harness_session
            self.interactive_calls += 1
            raise KeyboardInterrupt()

    app = _InterruptingAgentApp()
    request = _make_request(result_file=None, message=None)

    with pytest.raises(KeyboardInterrupt):
        await _run_cli_flow(app, request)

    captured = capsys.readouterr()
    assert app.interactive_calls == 2
    assert "Second Ctrl+C received; exiting fast-agent." in captured.err


@pytest.mark.asyncio
async def test_run_cli_flow_retries_interactive_after_cancelled_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _CancelledAgentApp(_DummyAgentApp):
        def __init__(self) -> None:
            super().__init__(["agent"])
            self.interactive_calls = 0

        async def interactive(
            self,
            agent_name: str | None = None,
            session_manager: object | None = None,
            harness_session: object | None = None,
        ) -> None:
            del agent_name, session_manager, harness_session
            self.interactive_calls += 1
            if self.interactive_calls == 1:
                raise asyncio.CancelledError()

    app = _CancelledAgentApp()
    request = _make_request(result_file=None, message=None)

    await _run_cli_flow(app, request)

    captured = capsys.readouterr()
    assert app.interactive_calls == 2
    assert "Interrupted operation; returning to fast-agent prompt." not in captured.err


def test_apply_shell_cwd_policy_preflight_interactive_honors_error_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fast = SimpleNamespace(
        agents={
            "agent": {
                "config": AgentConfig(
                    name="agent",
                    instruction="x",
                    servers=[],
                    shell=True,
                    cwd=Path("missing-shell-cwd"),
                )
            }
        },
        app=SimpleNamespace(
            context=SimpleNamespace(
                config=Settings(shell_execution=ShellSettings(missing_cwd_policy="error"))
            )
        ),
    )
    request = _make_request(result_file=None, message=None)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(typer.Exit) as exc_info:
        _apply_shell_cwd_policy_preflight(fast, request)

    assert exc_info.value.exit_code == 1
    assert "Shell cwd policy (error):" in capsys.readouterr().err


def test_apply_shell_cwd_policy_preflight_interactive_honors_warn_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notices: list[str] = []
    monkeypatch.setattr(
        "fast_agent.ui.enhanced_prompt.queue_startup_notice",
        notices.append,
    )
    fast = SimpleNamespace(
        agents={
            "agent": {
                "config": AgentConfig(
                    name="agent",
                    instruction="x",
                    servers=[],
                    shell=True,
                    cwd=Path("missing-shell-cwd"),
                )
            }
        },
        app=SimpleNamespace(
            context=SimpleNamespace(
                config=Settings(shell_execution=ShellSettings(missing_cwd_policy="warn"))
            )
        ),
    )
    request = _make_request(result_file=None, message=None)
    monkeypatch.chdir(tmp_path)

    _apply_shell_cwd_policy_preflight(fast, request)

    assert notices
    assert "Shell cwd policy (warn):" in notices[0]


def test_apply_shell_cwd_policy_preflight_interactive_honors_create_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fast = SimpleNamespace(
        agents={
            "agent": {
                "config": AgentConfig(
                    name="agent",
                    instruction="x",
                    servers=[],
                    shell=True,
                    cwd=Path("missing-shell-cwd"),
                )
            }
        },
        app=SimpleNamespace(
            context=SimpleNamespace(
                config=Settings(shell_execution=ShellSettings(missing_cwd_policy="create"))
            )
        ),
    )
    request = _make_request(result_file=None, message=None)
    monkeypatch.chdir(tmp_path)

    _apply_shell_cwd_policy_preflight(fast, request)

    assert (tmp_path / "missing-shell-cwd").is_dir()


def test_apply_shell_cwd_policy_preflight_interactive_create_errors_on_remaining_issue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    not_a_dir = tmp_path / "not-a-dir"
    not_a_dir.write_text("content", encoding="utf-8")
    fast = SimpleNamespace(
        agents={
            "agent": {
                "config": AgentConfig(
                    name="agent",
                    instruction="x",
                    servers=[],
                    shell=True,
                    cwd=Path("not-a-dir"),
                )
            }
        },
        app=SimpleNamespace(
            context=SimpleNamespace(
                config=Settings(shell_execution=ShellSettings(missing_cwd_policy="create"))
            )
        ),
    )
    request = _make_request(result_file=None, message=None)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(typer.Exit) as exc_info:
        _apply_shell_cwd_policy_preflight(fast, request)

    assert exc_info.value.exit_code == 1
    assert "Shell cwd policy (error):" in capsys.readouterr().err
