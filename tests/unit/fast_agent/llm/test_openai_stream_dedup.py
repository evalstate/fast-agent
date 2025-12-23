from fast_agent.llm.provider.openai.llm_openai import OpenAILLM


def test_extract_incremental_delta_with_cumulative_content() -> None:
    delta, cumulative = OpenAILLM._extract_incremental_delta("Hello, world", "")
    assert delta == "Hello, world"
    assert cumulative == "Hello, world"

    delta, cumulative = OpenAILLM._extract_incremental_delta("Hello, world!", cumulative)
    assert delta == "!"
    assert cumulative == "Hello, world!"


def test_extract_incremental_delta_with_non_cumulative_content() -> None:
    delta, cumulative = OpenAILLM._extract_incremental_delta("Part 1", "")
    assert delta == "Part 1"
    assert cumulative == "Part 1"

    delta, cumulative = OpenAILLM._extract_incremental_delta("Part 2", cumulative)
    assert delta == "Part 2"
    assert cumulative == "Part 1Part 2"
