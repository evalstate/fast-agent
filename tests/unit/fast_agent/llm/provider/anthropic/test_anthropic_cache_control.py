
from anthropic.types.beta import BetaMessageParam
from mcp.types import TextContent

from fast_agent.llm.provider.anthropic.cache_planner import AnthropicCachePlanner
from fast_agent.llm.provider.anthropic.llm_anthropic import AnthropicLLM
from fast_agent.llm.provider.anthropic.multipart_converter_anthropic import AnthropicConverter
from fast_agent.mcp.prompt_message_extended import PromptMessageExtended


def make_message(text: str, *, is_template: bool = False) -> PromptMessageExtended:
    return PromptMessageExtended(
        role="user", content=[TextContent(type="text", text=text)], is_template=is_template
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
    extended.extend(make_message(f"turn {i}") for i in range(6))

    plan_indices = planner.plan_indices(extended, cache_mode="auto", system_cache_blocks=system_cache_blocks)
    provider_msgs = [AnthropicConverter.convert_to_anthropic(msg) for msg in extended]
    for idx in plan_indices:
        AnthropicLLM._apply_cache_control_to_message(provider_msgs[idx])

    total_cache_blocks = system_cache_blocks + count_cache_controls(provider_msgs)

    assert total_cache_blocks <= 4
    assert len([i for i in plan_indices if i >= 2]) <= 1  # system + templates leave one slot


def test_conversation_cache_waits_for_walk_distance():
    planner = AnthropicCachePlanner(max_total_blocks=4)
    extended = [
        make_message("template", is_template=True),
        make_message("user 1"),
        make_message("assistant 1"),
    ]

    plan_indices = planner.plan_indices(extended, cache_mode="auto", system_cache_blocks=0)
    provider_msgs = [AnthropicConverter.convert_to_anthropic(msg) for msg in extended]

    assert plan_indices == [0]
    for idx in plan_indices:
        AnthropicLLM._apply_cache_control_to_message(provider_msgs[idx])

    assert count_cache_controls(provider_msgs) == 1
