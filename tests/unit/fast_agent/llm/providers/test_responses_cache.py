import pytest
from mcp_types import CallToolResult, ImageContent, TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.agents.tool_agent import ToolAgent
from fast_agent.agents.tool_runner import ToolRunner
from fast_agent.constants import FAST_AGENT_PENDING_MEDIA_ATTACHMENTS
from fast_agent.llm.provider.openai.responses import ResponsesLLM
from fast_agent.llm.provider.openai.responses_cache import (
    prepare_cached_request,
    read_cache_state,
)
from fast_agent.llm.request_params import RequestParams
from fast_agent.mcp.prompt import Prompt
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended
from fast_agent.mcp.prompt_serialization import from_json, to_json


def test_cache_keys_are_instance_scoped_not_explicit_breakpoints() -> None:
    first = ResponsesLLM(model="gpt-5.4")
    second = ResponsesLLM(model="gpt-5.4")
    params = RequestParams(model="gpt-5.4")
    before = first._build_response_args([], params, None)
    after = first._build_response_args([], params, None)
    independent = second._build_response_args([], params, None)
    assert before["prompt_cache_key"] == after["prompt_cache_key"]
    assert before["prompt_cache_key"] != independent["prompt_cache_key"]
    assert "prompt_cache_control" not in before


@pytest.mark.parametrize(
    "metadata",
    [
        {"truncation": "auto"},
        {"context_management": [{"type": "compaction", "compact_threshold": 10000}]},
        {"extra_body": {"truncation": "auto"}},
        {"extra_body": {"reasoning": {"effort": "high"}}},
        {"reasoning": {"mode": "pro"}},
        {"multi_agent": {"enabled": True}},
        {"extra_body": {"multi_agent": {"enabled": True}}},
        {"reasoning": {"effort": "none"}},
        {"reasoning": None},
        {"input": []},
        {"previous_response_id": "resp_external"},
    ],
)
def test_incompatible_astra_overrides_are_validated(metadata) -> None:
    llm = ResponsesLLM(model="gpt-6-astra")
    with pytest.raises(ValueError):
        prepare_cached_request(
            [Prompt.user("Work")],
            RequestParams(model="gpt-6-astra", metadata=metadata),
            key="key",
            default_effort="low",
            convert=llm._convert_message_to_items,
        )


@pytest.mark.parametrize("error_text", [False, True])
@pytest.mark.parametrize("pending_media", [False, True])
def test_tool_continuation_defers_effort_change_until_next_user(
    error_text: bool, pending_media: bool
) -> None:
    llm = ResponsesLLM(model="gpt-6-astra")
    history = [Prompt.user("Work")]

    def prepare(effort: str):
        return prepare_cached_request(
            history,
            RequestParams(model="gpt-6-astra"),
            key="key",
            default_effort=effort,
            convert=llm._convert_message_to_items,
        )

    _, state, _ = prepare("low")
    response = Prompt.assistant("Calling tool")
    state.attach(response)
    tool_message = PromptMessageExtended(
        role="user",
        content=[TextContent(type="text", text="Tool failed")] if error_text else [],
        tool_results={
            "call_test": CallToolResult(
                content=[TextContent(type="text", text="result")], is_error=error_text
            ),
        },
        channels={
            FAST_AGENT_PENDING_MEDIA_ATTACHMENTS: [
                ImageContent(type="image", data="abcd", mime_type="image/png")
            ]
        }
        if pending_media
        else None,
    )
    runner = ToolRunner(agent=ToolAgent(AgentConfig("media"), []), messages=[])
    runner._stage_tool_response(tool_message)
    history.extend([response, *runner.delta_messages])
    # Provenance must survive resuming midway through a tool continuation.
    history = from_json(to_json(history))
    items, state, params = prepare("high")
    assert not any(item["type"] == "configuration_update" for item in items)
    assert state.effort == "low"
    assert params.metadata and params.metadata["reasoning"]["effort"] == "low"
    response = Prompt.assistant("Tool finished")
    state.attach(response)
    history.extend([response, Prompt.user("Next task")])
    items, state, _ = prepare("high")
    assert items[-2] == {"type": "configuration_update", "reasoning": {"effort": "high"}}
    assert items[-1]["role"] == "user"
    assert state.effort == "high"

    response = Prompt.assistant("Done")
    state.attach(response)
    history.append(response)
    restored = from_json(to_json(history))
    assert read_cache_state(restored[-1]) == state
    replay, replay_state, _ = prepare_cached_request(
        [*restored, Prompt.user("Continue")],
        RequestParams(model="gpt-6-astra"),
        key="new-instance",
        default_effort="high",
        convert=llm._convert_message_to_items,
    )
    assert replay_state.effort == "high"
    update_indices = [
        index for index, item in enumerate(replay) if item["type"] == "configuration_update"
    ]
    assert len(update_indices) == 1
    assert replay[update_indices[0] + 1] == items[-1]


def test_empty_replacement_history_establishes_new_baseline() -> None:
    llm = ResponsesLLM(model="gpt-6-astra")
    for effort in ("low", "high"):
        items, state, params = prepare_cached_request(
            [Prompt.user("Summary or fresh conversation")],
            RequestParams(model="gpt-6-astra"),
            key="key",
            default_effort=effort,
            convert=llm._convert_message_to_items,
        )
        assert state.effort == effort
        assert params.metadata and params.metadata["reasoning"]["effort"] == effort
        assert not any(item["type"] == "configuration_update" for item in items)
