from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from mcp.types import TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.cli.runtime.agent_setup import (
    _apply_shell_cwd_policy_preflight,
    _build_fan_out_result_paths,
    _build_result_file_with_suffix,
    _export_result_histories,
    _find_last_assistant_text,
    _run_single_agent_cli_flow,
    _sanitize_result_suffix,
)
from fast_agent.cli.runtime.run_request import AgentRunRequest
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


class _DummyAgent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.message_history: list[object] = []


class _DummyAgentApp:
    def __init__(self, agent_names: list[str], *, default_agent: str | None = None) -> None:
        self._agents = {name: _DummyAgent(name) for name in agent_names}
        self._default_agent = default_agent or agent_names[0]

    def _agent(self, agent_name: str | None):
        if agent_name is None:
            return self._agents[self._default_agent]
        return self._agents[agent_name]

    async def interactive(self, agent_name: str | None = None) -> None:
        del agent_name

    async def send(self, message: str, agent_name: str | None = None) -> str:
        del agent_name
        return message


def _make_request(
    *,
    result_file: str | None,
    target_agent_name: str | None = None,
    message: str | None = "hello",
    prompt_file: str | None = None,
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
    )


def test_build_result_file_with_suffix_without_extension() -> None:
    assert _build_result_file_with_suffix(Path("foo"), "haiku35") == Path("foo-haiku35")


def test_sanitize_result_suffix() -> None:
    assert _sanitize_result_suffix("openai/gpt-4o") == "openai_gpt-4o"
    assert _sanitize_result_suffix("  model name  ") == "model_name"


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
async def test_run_single_agent_cli_flow_retries_interactive_after_keyboard_interrupt(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _InterruptingAgentApp(_DummyAgentApp):
        def __init__(self) -> None:
            super().__init__(["agent"])
            self.interactive_calls = 0

        async def interactive(self, agent_name: str | None = None) -> None:
            del agent_name
            self.interactive_calls += 1
            if self.interactive_calls == 1:
                raise KeyboardInterrupt()

    app = _InterruptingAgentApp()
    request = _make_request(result_file=None, message=None)

    await _run_single_agent_cli_flow(app, request)

    captured = capsys.readouterr()
    assert app.interactive_calls == 2
    assert "Interrupted operation; returning to fast-agent prompt." in captured.err


@pytest.mark.asyncio
async def test_run_single_agent_cli_flow_exits_after_double_keyboard_interrupt(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _InterruptingAgentApp(_DummyAgentApp):
        def __init__(self) -> None:
            super().__init__(["agent"])
            self.interactive_calls = 0

        async def interactive(self, agent_name: str | None = None) -> None:
            del agent_name
            self.interactive_calls += 1
            raise KeyboardInterrupt()

    app = _InterruptingAgentApp()
    request = _make_request(result_file=None, message=None)

    with pytest.raises(KeyboardInterrupt):
        await _run_single_agent_cli_flow(app, request)

    captured = capsys.readouterr()
    assert app.interactive_calls == 2
    assert "Second Ctrl+C received; exiting fast-agent." in captured.err


@pytest.mark.asyncio
async def test_run_single_agent_cli_flow_retries_interactive_after_cancelled_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _CancelledAgentApp(_DummyAgentApp):
        def __init__(self) -> None:
            super().__init__(["agent"])
            self.interactive_calls = 0

        async def interactive(self, agent_name: str | None = None) -> None:
            del agent_name
            self.interactive_calls += 1
            if self.interactive_calls == 1:
                raise asyncio.CancelledError()

    app = _CancelledAgentApp()
    request = _make_request(result_file=None, message=None)

    await _run_single_agent_cli_flow(app, request)

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
                config=SimpleNamespace(
                    shell_execution=SimpleNamespace(missing_cwd_policy="error")
                )
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
                config=SimpleNamespace(
                    shell_execution=SimpleNamespace(missing_cwd_policy="warn")
                )
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
                config=SimpleNamespace(
                    shell_execution=SimpleNamespace(missing_cwd_policy="create")
                )
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
                config=SimpleNamespace(
                    shell_execution=SimpleNamespace(missing_cwd_policy="create")
                )
            )
        ),
    )
    request = _make_request(result_file=None, message=None)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(typer.Exit) as exc_info:
        _apply_shell_cwd_policy_preflight(fast, request)

    assert exc_info.value.exit_code == 1
    assert "Shell cwd policy (error):" in capsys.readouterr().err
