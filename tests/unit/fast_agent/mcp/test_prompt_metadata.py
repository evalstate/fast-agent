from mcp.types import GetPromptResult

from fast_agent.mcp.prompt_metadata import (
    prompt_arguments,
    prompt_display_name,
    with_prompt_metadata,
)


def _result_with_meta(meta: object) -> GetPromptResult:
    return GetPromptResult(messages=[]).model_copy(update={"meta": meta})


def test_prompt_metadata_helpers_ignore_malformed_meta() -> None:
    result = _result_with_meta("not a mapping")

    assert prompt_display_name(result, "fallback") == "fallback"
    assert prompt_arguments(result) is None


def test_with_prompt_metadata_replaces_malformed_meta() -> None:
    result = with_prompt_metadata(
        _result_with_meta("not a mapping"),
        namespaced_name="server/prompt",
        arguments={"name": "Ada"},
    )

    assert prompt_display_name(result, "fallback") == "server/prompt"
    assert prompt_arguments(result) == {"name": "Ada"}
