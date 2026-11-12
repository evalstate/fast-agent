import pytest

from fast_agent.llm.provider.bedrock.llm_bedrock import BedrockLLM


@pytest.mark.unit
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        pytest.param(
            '{"name": "get_weather", "arguments": {"city": "Paris"}}',
            [("get_weather", {"city": "Paris"})],
            id="object-arguments",
        ),
        pytest.param(
            '{"name": "ping", "arguments": {}}',
            [("ping", {})],
            id="empty-object-arguments",
        ),
        pytest.param(
            'I will check.\n{"name": "get_weather", "arguments": {"city": "Paris"}}',
            [("get_weather", {"city": "Paris"})],
            id="text-before-call",
        ),
        pytest.param(
            '{"name": "search", "arguments": {"filter": {"tags": ["a"]}, "query": "a}b"}}',
            [("search", {"filter": {"tags": ["a"]}, "query": "a}b"})],
            id="nested-arguments-and-brace-in-string",
        ),
        pytest.param(
            '{"name": "get_weather", "arguments": "Paris"}',
            [("get_weather", {})],
            id="non-object-arguments",
        ),
        pytest.param('{"name": "Paris"}', [], id="object-without-arguments"),
    ],
)
def test_system_prompt_single_json_tool_call(
    text: str, expected: list[tuple[str, dict[str, object]]]
) -> None:
    llm = BedrockLLM(model="test.model")

    calls = llm._parse_system_prompt_tool_response({"content": [{"text": text}]}, "test.model")

    assert [(call["name"], call["arguments"]) for call in calls] == expected
