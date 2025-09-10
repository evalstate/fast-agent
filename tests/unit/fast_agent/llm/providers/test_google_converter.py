from google.genai import types
from mcp.types import CallToolResult, TextContent

from fast_agent.llm.provider.google.google_converter import GoogleConverter


def test_convert_function_results_to_google_text_only():
    converter = GoogleConverter()

    # Create a simple text-only tool result
    result = CallToolResult(
        content=[TextContent(type="text", text="Weather is sunny")], isError=False
    )

    contents = converter.convert_function_results_to_google([("weather", result)])

    # One google Content with role 'tool'
    assert isinstance(contents, list)
    assert len(contents) == 1
    content = contents[0]
    assert isinstance(content, types.Content)
    assert content.role == "tool"
    assert content.parts
    # First part should be a function response named 'weather'
    fn_resp = content.parts[0].function_response
    assert fn_resp is not None
    assert fn_resp.name == "weather"
    assert isinstance(fn_resp.response, dict)
    assert fn_resp.response.get("tool_name") == "weather"
