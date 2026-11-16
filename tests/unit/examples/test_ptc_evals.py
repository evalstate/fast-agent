"""The example checker follows the runtime's sandbox name and parent call IDs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from examples.ptc import evals
from fast_agent.constants import FAST_AGENT_PTC_CALLS
from fast_agent.ptc import PTC_TOOL_NAME

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(params=["code", "input"])
def history(request: pytest.FixtureRequest) -> evals.History:
    return [
        {
            "tool_calls": {
                "batch": {
                    "params": {
                        "name": PTC_TOOL_NAME,
                        "arguments": {
                            request.param: "await asyncio.gather(orders__get_orders(), "
                            "orders__get_orders())"
                        },
                    }
                }
            }
        },
        {
            "tool_results": {
                "batch": {
                    "content": [{"type": "image"}],
                    "_meta": {
                        FAST_AGENT_PTC_CALLS: [
                            {"tool": "orders__get_orders"},
                            {"tool": "orders__get_orders"},
                        ]
                    },
                }
            }
        },
    ]


def test_checker_recognizes_current_sandbox_history(history: evals.History) -> None:
    assert evals.batched_orders(history) is None
    assert evals.image_in_code_result(history) is None
    assert evals.nested_tools(history) == ["orders__get_orders", "orders__get_orders"]


def test_checker_ignores_other_top_level_tools(history: evals.History) -> None:
    history[0]["tool_calls"]["batch"]["params"]["name"] = "orders__get_orders"

    assert evals.batched_orders(history) is not None
    assert evals.image_in_code_result(history) is not None
    assert evals.nested_tools(history) == []


@pytest.mark.parametrize("position", ["before", "alongside", "after"])
@pytest.mark.parametrize("name", ["orders__list_customers", "orders__get_orders", "read_text_file"])
def test_case_rejects_direct_calls_even_with_successful_code_path(
    history: evals.History, position: str, name: str
) -> None:
    direct = {"direct": {"params": {"name": name, "arguments": {}}}}
    if position == "alongside":
        history[0]["tool_calls"].update(direct)
    else:
        history.insert(0 if position == "before" else len(history), {"tool_calls": direct})

    assert evals.batched_orders(history) is None
    reason = evals.CASES[0].check(history)
    assert reason is not None and name in reason


def test_case_allows_deliberately_direct_slash_command(history: evals.History) -> None:
    history[0]["tool_calls"]["status"] = {"params": {"name": "slash_command"}}
    assert evals.CASES[0].check(history) is None


def test_batch_dispatches_must_belong_to_the_gather_invocation(history: evals.History) -> None:
    results = history[1]["tool_results"]
    results["other"] = results.pop("batch")

    assert evals.batched_orders(history) is not None


def test_failure_messages_use_current_sandbox_name() -> None:
    for check in (evals.batched_orders, evals.image_in_code_result, evals.direct_slash_command):
        reason = check([])
        assert reason is not None
        assert PTC_TOOL_NAME in reason


@pytest.mark.parametrize("mode", ["profile", "canonical"])
def test_eval_arm_controls_loaded_example_setting(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fast_agent.config import get_settings

    args = evals.command(evals.CASES[0], None, "passthrough", mode)
    assert args[:2] == ["env", f"PTC__FUNCTION_NAMES={mode}"]
    monkeypatch.setenv("PTC__FUNCTION_NAMES", mode)
    settings = get_settings(evals.HERE / "fast-agent.yaml", no_home=True)
    assert settings.ptc.function_names == mode
    card = (evals.HERE / "agent-cards/ptc-eval.md").read_text()
    assert "Canonical PTC uses" not in card


@pytest.mark.asyncio
@pytest.mark.parametrize("actual", [None, 0, 1])
async def test_eval_checks_and_prints_actual_vocabulary_without_code_calls(
    actual: int | None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fast_agent.ptc.vocabulary import PTC_VOCABULARY_META

    case = evals.Case("no-uptake", "hello", ("hello",), lambda history: None)
    (tmp_path / "no-uptake.json").write_text(
        json.dumps(
            {
                "messages": [{"role": "assistant", "content": [{"text": "hello"}]}],
                "metadata": {PTC_VOCABULARY_META: actual, "ptc_functions": ["bash"]},
            }
        )
    )
    child = AsyncMock()
    child.returncode = 0
    child.communicate.return_value = (b"", None)
    monkeypatch.setattr(evals.asyncio, "create_subprocess_exec", AsyncMock(return_value=child))
    passed, detail, _ = await evals.run_case(case, "passthrough", tmp_path, "profile")
    assert passed is (actual == 0), detail
    assert f"vocabulary={actual}, functions=bash" in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["canonical", "profile"])
@pytest.mark.parametrize("transient", [False, True])
async def test_results_header_records_actual_surface_without_tool_calls(
    mode: str,
    transient: bool,
    tmp_path: Path,
) -> None:
    from fast_agent.agents.agent_types import AgentConfig
    from fast_agent.agents.mcp_agent import McpAgent
    from fast_agent.cli.runtime.agent_setup import _save_result_history
    from fast_agent.config import Settings
    from fast_agent.context import Context
    from fast_agent.core.agent_app import AgentApp
    from fast_agent.mcp.prompt_serialization import load_messages
    from fast_agent.ptc.vocabulary import PTC_VOCABULARY_META
    from fast_agent.types import PromptMessageExtended

    agent = McpAgent(
        AgentConfig(
            name="export", servers=[], shell=True, model=f"passthrough?ptc=on&ptc_names={mode}"
        ),
        context=Context(config=Settings()),
    )
    path = tmp_path / "result.json"
    messages = [PromptMessageExtended(role="assistant", content=[])] if transient else None
    try:
        await _save_result_history(
            AgentApp({"export": agent}),
            agent_name="export",
            output_path=path,
            messages_override=messages,
        )
        document = json.loads(path.read_text())
        assert document["metadata"][PTC_VOCABULARY_META] == (1 if mode == "canonical" else 0)
        assert ("shell" if mode == "canonical" else "bash") in document["metadata"]["ptc_functions"]
        assert load_messages(str(path)) == (messages or [])
    finally:
        await agent.shutdown()
