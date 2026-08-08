from types import SimpleNamespace

from fast_agent.agents.tool_call_planning import plan_tool_calls


def _tool_request(name: str, arguments: dict[str, object] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        params=SimpleNamespace(
            name=name,
            arguments=arguments,
        )
    )


def test_plan_tool_calls_resolves_unique_case_only_name() -> None:
    plan = plan_tool_calls(
        [("call-1", _tool_request("shell", {"command": "pwd"}))],
        available_tools=["Shell"],
        execution_tools={},
    )

    assert plan.unavailable_calls == []
    assert len(plan.planned_calls) == 1
    assert plan.planned_calls[0].name == "Shell"
    assert plan.planned_calls[0].arguments == {"command": "pwd"}


def test_plan_tool_calls_keeps_valid_siblings_when_one_name_is_unavailable() -> None:
    plan = plan_tool_calls(
        [
            ("call-1", _tool_request("missing_tool")),
            ("call-2", _tool_request("Shell", {"command": "pwd"})),
        ],
        available_tools=["Shell"],
        execution_tools={},
    )

    assert [(call.correlation_id, call.name) for call in plan.planned_calls] == [
        ("call-2", "Shell")
    ]
    assert [(call.correlation_id, call.name) for call in plan.unavailable_calls] == [
        ("call-1", "missing_tool")
    ]


def test_plan_tool_calls_does_not_resolve_ambiguous_case_only_name() -> None:
    plan = plan_tool_calls(
        [("call-1", _tool_request("SHELL"))],
        available_tools=["Shell", "shell"],
        execution_tools={},
    )

    assert plan.planned_calls == []
    assert len(plan.unavailable_calls) == 1
    assert plan.unavailable_calls[0].name == "SHELL"
