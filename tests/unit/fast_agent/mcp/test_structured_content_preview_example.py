from pathlib import Path
from runpy import run_path

from mcp_types import CallToolResult


def test_structured_content_preview_builds_sdk_v2_result() -> None:
    example = (
        Path(__file__).parents[4]
        / "examples"
        / "mcp"
        / "structured-content-preview"
        / "preview_server.py"
    )
    namespace = run_path(str(example), run_name="preview_server")
    tool_result = namespace["_tool_result"]

    result = tool_result(
        text_payloads=[{"ticket_id": "T-100"}],
        structured_payload={"tickets": [{"ticket_id": "T-100"}]},
    )

    assert isinstance(result, CallToolResult)
    assert result.structured_content == {"tickets": [{"ticket_id": "T-100"}]}
