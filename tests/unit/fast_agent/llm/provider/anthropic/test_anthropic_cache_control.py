from typing import Literal

from anthropic.types.beta import (
    BetaMessageParam,
    BetaRedactedThinkingBlockParam,
    BetaTextBlockParam,
    BetaThinkingBlockParam,
)
from mcp.types import TextContent

from fast_agent.llm.provider.anthropic.cache_planner import AnthropicCachePlanner
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM
from fast_agent.llm.provider.anthropic.multipart_converter_anthropic import AnthropicConverter
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


def make_message(
    text: str,
    *,
    role: Literal["user", "assistant"] = "user",
    is_template: bool = False,
) -> PromptMessageExtended:
    return PromptMessageExtended(
        role=role, content=[TextContent(type="text", text=text)], is_template=is_template
    )


def count_cache_controls(messages: list[BetaMessageParam]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, str):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("cache_control"):
                total += 1
    return total


def test_template_cache_respects_budget():
    planner = AnthropicCachePlanner(max_total_blocks=4)
    extended = [
        make_message("template 1", is_template=True),
        make_message("template 2", is_template=True),
        make_message("user turn"),
    ]

    plan_indices = planner.plan_indices(extended, cache_mode="prompt", system_cache_blocks=0)
    provider_msgs = [AnthropicConverter.convert_to_anthropic(msg) for msg in extended]

    for idx in plan_indices:
        AnthropicLLM._apply_cache_control_to_message(provider_msgs[idx])

    first_blocks = list(provider_msgs[0]["content"])
    second_blocks = list(provider_msgs[1]["content"])
    assert "cache_control" in first_blocks[-1]
    assert "cache_control" in second_blocks[-1]


def test_conversation_cache_respects_four_block_limit():
    planner = AnthropicCachePlanner(max_total_blocks=4)
    system_cache_blocks = 1
    extended = [
        make_message("template 1", is_template=True),
        make_message("template 2", is_template=True),
    ]
    extended.extend(
        [
            make_message("user 1"),
            make_message("assistant 1", role="assistant"),
            make_message("user 2"),
            make_message("assistant 2", role="assistant"),
            make_message("user 3"),
            make_message("assistant 3", role="assistant"),
        ]
    )

    plan_indices = planner.plan_indices(
        extended, cache_mode="auto", system_cache_blocks=system_cache_blocks
    )
    provider_msgs = [AnthropicConverter.convert_to_anthropic(msg) for msg in extended]
    for idx in plan_indices:
        AnthropicLLM._apply_cache_control_to_message(provider_msgs[idx])

    total_cache_blocks = system_cache_blocks + count_cache_controls(provider_msgs)

    assert total_cache_blocks <= 4
    assert plan_indices == [0, 1, 7]


def test_conversation_cache_checkpoints_latest_assistant_on_next_request():
    planner = AnthropicCachePlanner(max_total_blocks=4)
    extended = [
        make_message("template", is_template=True),
        make_message("user 1"),
        make_message("assistant 1", role="assistant"),
        make_message("tool result"),
    ]

    plan_indices = planner.plan_indices(extended, cache_mode="auto", system_cache_blocks=0)
    provider_msgs = [AnthropicConverter.convert_to_anthropic(msg) for msg in extended]

    assert plan_indices == [0, 2]
    for idx in plan_indices:
        AnthropicLLM._apply_cache_control_to_message(provider_msgs[idx])

    assert count_cache_controls(provider_msgs) == 2


def test_conversation_cache_reapplies_previous_assistant_checkpoint():
    planner = AnthropicCachePlanner(max_total_blocks=4)
    extended = [
        make_message("user 1"),
        make_message("assistant 1", role="assistant"),
        make_message("tool result 1"),
        make_message("assistant 2", role="assistant"),
        make_message("tool result 2"),
    ]

    assert planner.plan_indices(
        extended,
        cache_mode="auto",
        system_cache_blocks=1,
    ) == [1, 3]


def test_conversation_cache_reserves_marker_after_large_template_prefix():
    planner = AnthropicCachePlanner(max_total_blocks=4)
    extended = [
        make_message("template 1", is_template=True),
        make_message("template 2", is_template=True),
        make_message("template 3", is_template=True),
        make_message("user 1"),
        make_message("assistant 1", role="assistant"),
        make_message("tool result"),
    ]

    assert planner.plan_indices(
        extended,
        cache_mode="auto",
        system_cache_blocks=1,
    ) == [0, 1, 4]


def test_process_poll_boundary_replaces_periodic_conversation_markers():
    planner = AnthropicCachePlanner(max_total_blocks=4)
    extended = [make_message(f"turn {i}") for i in range(8)]

    plan_indices = planner.plan_indices(
        extended,
        cache_mode="auto",
        system_cache_blocks=1,
        process_poll_boundary=3,
    )

    assert plan_indices == [3]


def test_process_poll_boundary_shares_budget_with_template_prefix():
    planner = AnthropicCachePlanner(max_total_blocks=4)
    extended = [
        make_message("template 1", is_template=True),
        make_message("template 2", is_template=True),
        make_message("user"),
        make_message("assistant", role="assistant"),
        make_message("execute result"),
        make_message("poll request", role="assistant"),
        make_message("poll result"),
    ]

    assert planner.plan_indices(
        extended,
        cache_mode="auto",
        system_cache_blocks=1,
        process_poll_boundary=4,
    ) == [0, 1, 4]


def test_cache_control_skips_thinking_blocks():
    message = BetaMessageParam(
        role="assistant",
        content=[
            BetaThinkingBlockParam(
                type="thinking",
                thinking="reasoning",
                signature="signature",
            ),
            BetaTextBlockParam(type="text", text="answer"),
        ],
    )

    assert AnthropicLLM._apply_cache_control_to_message(message)
    thinking, text = [
        {str(key): value for key, value in block.items()}
        for block in message["content"]
        if isinstance(block, dict)
    ]
    assert "cache_control" not in thinking
    assert text["cache_control"] == {"type": "ephemeral", "ttl": "5m"}


def test_cache_control_rejects_thinking_only_message():
    message = BetaMessageParam(
        role="assistant",
        content=[
            BetaThinkingBlockParam(
                type="thinking",
                thinking="reasoning",
                signature="signature",
            ),
            BetaRedactedThinkingBlockParam(
                type="redacted_thinking",
                data="redacted",
            ),
        ],
    )

    assert not AnthropicLLM._apply_cache_control_to_message(message)
    assert all(
        "cache_control" not in block
        for block in message["content"]
        if isinstance(block, dict)
    )
