"""The example checker follows the runtime's sandbox name and parent call IDs."""

from __future__ import annotations

import pytest

from examples.ptc import evals
from fast_agent.constants import FAST_AGENT_PTC_CALLS
from fast_agent.ptc import PTC_TOOL_NAME


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


def test_batch_dispatches_must_belong_to_the_gather_invocation(history: evals.History) -> None:
    results = history[1]["tool_results"]
    results["other"] = results.pop("batch")

    assert evals.batched_orders(history) is not None


def test_failure_messages_use_current_sandbox_name() -> None:
    for check in (evals.batched_orders, evals.image_in_code_result, evals.direct_slash_command):
        reason = check([])
        assert reason is not None
        assert PTC_TOOL_NAME in reason
