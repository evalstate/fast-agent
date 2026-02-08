from __future__ import annotations

from pathlib import Path

import pytest
import typer

from fast_agent.cli.runtime.agent_setup import (
    _build_fan_out_result_paths,
    _build_result_file_with_suffix,
    _export_result_histories,
    _sanitize_result_suffix,
)
from fast_agent.cli.runtime.run_request import AgentRunRequest


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


def _make_request(*, result_file: str | None, target_agent_name: str | None = None) -> AgentRunRequest:
    return AgentRunRequest(
        name="test",
        instruction="instruction",
        config_path=None,
        server_list=None,
        agent_cards=None,
        card_tools=None,
        model=None,
        message="hello",
        prompt_file=None,
        result_file=result_file,
        resume=None,
        url_servers=None,
        stdio_servers=None,
        agent_name="agent",
        target_agent_name=target_agent_name,
        skills_directory=None,
        environment_dir=None,
        noenv=False,
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
