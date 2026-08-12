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
        known_tool_names=["Shell"],
        case_insensitive_tool_names=["Shell"],
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
        known_tool_names=["Shell"],
        case_insensitive_tool_names=["Shell"],
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
        known_tool_names=["Shell", "shell"],
        case_insensitive_tool_names=["Shell", "shell"],
    )

    assert plan.planned_calls == []
    assert len(plan.unavailable_calls) == 1
    assert plan.unavailable_calls[0].name == "SHELL"


def test_plan_tool_calls_casefolds_only_model_visible_names() -> None:
    plan = plan_tool_calls(
        [("call-1", _tool_request("WRITE_TEXT_FILE"))],
        known_tool_names=["write_text_file", "Write_Text_File"],
        case_insensitive_tool_names=["write_text_file"],
    )

    assert plan.unavailable_calls == []
    assert plan.planned_calls[0].name == "write_text_file"
