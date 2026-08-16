from mcp_types import CallToolResult, TextContent

from fast_agent.constants import FAST_AGENT_SUBAGENT_RESULT_METADATA
from fast_agent.ui.subagent_result_presentation import (
    SubagentResultPresentation,
    build_subagent_result_presentation,
)


def test_build_subagent_result_presentation_uses_result_metadata() -> None:
    result = CallToolResult(
        content=[TextContent(type="text", text="done")],
        meta={
            FAST_AGENT_SUBAGENT_RESULT_METADATA: {
                "alias": "01_audit",
                "label": "Audit",
                "child_agent_name": "child",
                "model_spec": "provider.model",
                "child_session_id": "session-1",
            }
        },
    )

    presentation = build_subagent_result_presentation(result)

    assert presentation == SubagentResultPresentation(
        message_text="done",
        name="subagent: 01_audit",
        model="provider.model",
        bottom_items=["session session-1"],
        highlight_indexes=[0],
    )


def test_build_subagent_result_presentation_handles_missing_metadata() -> None:
    result = CallToolResult(content=[TextContent(type="text", text="failed")], is_error=True)

    presentation = build_subagent_result_presentation(result)

    assert presentation == SubagentResultPresentation(
        message_text="failed",
        name="subagent",
        model=None,
        bottom_items=None,
        highlight_indexes=None,
    )
