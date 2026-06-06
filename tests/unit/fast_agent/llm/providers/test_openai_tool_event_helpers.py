from types import SimpleNamespace

from fast_agent.llm.provider.openai.tool_event_helpers import (
    fallback_tool_spec,
    first_nonempty_string,
    item_is_responses_tool,
    item_type_is_responses_function_tool_call,
    responses_lifecycle_event_info,
    responses_tool_name,
    responses_tool_use_id,
    tool_family_for_item_type,
    tool_stream_log_record,
)


def test_responses_tool_name_uses_fixed_names_for_provider_tool_types() -> None:
    assert responses_tool_name(SimpleNamespace(type="tool_search_call")) == "tool_search"
    assert responses_tool_name(SimpleNamespace(type="web_search_call")) == "web_search"


def test_responses_tool_classifiers_ignore_non_string_types() -> None:
    assert item_is_responses_tool(SimpleNamespace(type=[])) is False
    assert item_type_is_responses_function_tool_call([]) is False


def test_first_nonempty_string_strips_and_skips_non_strings() -> None:
    assert first_nonempty_string(None, 123, "  value  ") == "value"
    assert first_nonempty_string("", "   ", object()) is None


def test_responses_tool_name_qualifies_mcp_tools_with_server_label() -> None:
    assert (
        responses_tool_name(
            SimpleNamespace(type="mcp_call", server_label="stripe", name="create_link")
        )
        == "stripe/create_link"
    )
    assert (
        responses_tool_name(SimpleNamespace(type="mcp_list_tools", server_label="stripe"))
        == "stripe/mcp_list_tools"
    )


def test_responses_tool_name_uses_mcp_tool_name_fallback() -> None:
    assert (
        responses_tool_name(
            SimpleNamespace(type="mcp_call", server_label="stripe", tool_name="refund")
        )
        == "stripe/refund"
    )
    assert responses_tool_name(SimpleNamespace(type="mcp_call")) == "mcp_call"


def test_responses_tool_name_ignores_non_string_sdk_name_values() -> None:
    assert responses_tool_name(SimpleNamespace(type="function_call", name=123)) == "tool"
    assert (
        responses_tool_name(SimpleNamespace(type="mcp_call", server_label="stripe", name=123))
        == "stripe/mcp_call"
    )


def test_responses_tool_name_strips_and_ignores_blank_sdk_name_values() -> None:
    assert responses_tool_name(SimpleNamespace(type="function_call", name="  lookup  ")) == "lookup"
    assert responses_tool_name(SimpleNamespace(type="function_call", name="   ")) == "tool"
    assert (
        responses_tool_name(
            SimpleNamespace(type="mcp_call", server_label="  stripe  ", name="  pay  ")
        )
        == "stripe/pay"
    )
    assert (
        responses_tool_name(
            SimpleNamespace(type="mcp_call", server_label="   ", tool_name="  refund  ")
        )
        == "refund"
    )


def test_tool_family_for_item_type_uses_default_tool_family() -> None:
    assert tool_family_for_item_type("web_search_call") == "web_search"
    assert tool_family_for_item_type("mcp_call") == "remote_tool"
    assert tool_family_for_item_type("function_call") == "tool"
    assert tool_family_for_item_type(None) == "tool"


def test_item_type_is_responses_function_tool_call_matches_function_call_items() -> None:
    assert item_type_is_responses_function_tool_call("function_call")
    assert item_type_is_responses_function_tool_call("custom_tool_call")
    assert not item_type_is_responses_function_tool_call("function_call_output")
    assert not item_type_is_responses_function_tool_call("mcp_call")
    assert not item_type_is_responses_function_tool_call(None)


def test_responses_lifecycle_event_info_classifies_statuses() -> None:
    cases = {
        "response.web_search_call.searching": (
            "web_search_call",
            "web_search",
            "searching",
            "start",
        ),
        "response.web_search_call.completed": (
            "web_search_call",
            "web_search",
            "completed",
            "stop",
        ),
        "response.mcp_list_tools.in_progress": (
            "mcp_list_tools",
            "mcp_list_tools",
            "in_progress",
            "start",
        ),
        "response.mcp_call.failed": (
            "mcp_call",
            "mcp_call",
            "failed",
            "stop",
        ),
        "response.tool_search_call.queued": (
            "tool_search_call",
            "tool_search",
            "queued",
            "start",
        ),
    }

    for event_type, expected in cases.items():
        info = responses_lifecycle_event_info(event_type)
        assert info is not None
        assert (info.item_type, info.tool_name, info.status, info.lifecycle) == expected


def test_responses_lifecycle_event_info_ignores_non_lifecycle_events() -> None:
    assert responses_lifecycle_event_info(None) is None
    assert responses_lifecycle_event_info("response.function_call.in_progress") is None
    assert responses_lifecycle_event_info("response.output_item.done") is None


def test_responses_lifecycle_event_info_can_include_function_call_statuses() -> None:
    info = responses_lifecycle_event_info(
        "response.function_call_arguments.done",
        include_function_calls=True,
    )

    assert info is not None
    assert (info.item_type, info.tool_name, info.status, info.lifecycle) == (
        "function_call_arguments",
        "tool",
        "done",
        "status",
    )


def test_tool_stream_log_record_marks_terminal_and_fallback_events() -> None:
    message, data = tool_stream_log_record(
        agent_name="agent",
        model="model",
        tool_name="search",
        tool_use_id="call-1",
        event_type="stop",
        fallback=True,
    )

    assert message == "Model emitted fallback tool notification"
    assert data["tool_terminal"] is True
    assert data["fallback"] is True
    assert data["tool_event"] == "stop"


def test_fallback_tool_spec_uses_canonical_provider_tool_helpers() -> None:
    cases = [
        (
            SimpleNamespace(type="tool_search_call", id="ts_1", call_id="call_1"),
            ("tool_search", "call_1", "remote_tool_search"),
        ),
        (
            SimpleNamespace(type="web_search_call", id="ws_1"),
            ("web_search", "ws_1", "web_search"),
        ),
        (
            SimpleNamespace(type="mcp_list_tools", id="mlt_1", server_label="stripe"),
            ("stripe/mcp_list_tools", "mlt_1", "remote_tool_listing"),
        ),
        (
            SimpleNamespace(type="mcp_call", id="mcp_1", server_label="stripe", name="pay"),
            ("stripe/pay", "mcp_1", "remote_tool"),
        ),
        (
            SimpleNamespace(type="function_call", call_id="fn_1", name="lookup"),
            ("lookup", "fn_1", "tool"),
        ),
    ]

    for item, expected in cases:
        assert fallback_tool_spec(item, index=3) == expected


def test_responses_tool_use_id_ignores_blank_and_non_string_provider_ids() -> None:
    item = SimpleNamespace(type="function_call", call_id=123, id="  fc_1  ")

    assert responses_tool_use_id(item, index=4) == "fc_1"
    assert (
        responses_tool_use_id(
            SimpleNamespace(type="function_call", call_id="", id=123),
            index=4,
            item_id=" item_1 ",
        )
        == "item_1"
    )
    assert (
        responses_tool_use_id(SimpleNamespace(type="function_call", call_id="", id=123), index=4)
        == "function_call-4"
    )
