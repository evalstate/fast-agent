from fast_agent.tool_activity_presentation import (
    PRESERVE_SECTION_TOOL_FAMILIES,
    REMOTE_STATUS_TOOL_FAMILIES,
    TOOL_ACTIVITY_FAMILIES,
    build_tool_activity_presentation,
    classify_tool_activity_family,
    is_tool_activity_family,
    tool_activity_family_uses_status_body,
    tool_activity_status_text,
)


def test_tool_activity_family_groups_are_valid_families() -> None:
    valid_families = frozenset(TOOL_ACTIVITY_FAMILIES)

    assert set(REMOTE_STATUS_TOOL_FAMILIES) <= valid_families
    assert set(PRESERVE_SECTION_TOOL_FAMILIES) <= valid_families


def test_build_tool_activity_presentation_uses_family_labels() -> None:
    remote = build_tool_activity_presentation(
        tool_name="docs/search",
        family="remote_tool",
        phase="call",
    )
    listing = build_tool_activity_presentation(
        tool_name="mcp_list_tools",
        family="remote_tool_listing",
        phase="result",
    )

    assert remote.display_name == "remote tool: search"
    assert remote.type_label == "remote tool call"
    assert remote.preserve_sections is True
    assert listing.display_name == "Loading remote tools"
    assert listing.type_label is None
    assert listing.preserve_sections is False


def test_tool_activity_status_text_uses_family_specific_copy() -> None:
    assert (
        tool_activity_status_text(family="remote_tool_search", status="in_progress")
        == "searching deferred tools..."
    )
    assert (
        tool_activity_status_text(family="remote_tool_listing", status="completed")
        == "remote tools loaded"
    )
    assert (
        tool_activity_status_text(family="remote_tool", status="failed")
        == "remote tool call failed"
    )
    assert tool_activity_status_text(family="tool", status=" QUEUED ") == "queued..."
    assert tool_activity_status_text(family="web_search", status="unknown_status") == "unknown status"


def test_tool_activity_family_predicates_capture_shared_rendering_groups() -> None:
    assert is_tool_activity_family("remote_tool_search")
    assert not is_tool_activity_family("unknown")
    assert tool_activity_family_uses_status_body("remote_tool_search")
    assert tool_activity_family_uses_status_body("remote_tool_listing")
    assert not tool_activity_family_uses_status_body("remote_tool")
    assert not tool_activity_family_uses_status_body(None)


def test_classify_tool_activity_family_handles_provider_names() -> None:
    assert (
        classify_tool_activity_family(tool_name="tool_search", provider_tool_type="tool_search_call")
        == "remote_tool_search"
    )
    assert (
        classify_tool_activity_family(tool_name="server/mcp_list_tools")
        == "remote_tool_listing"
    )
    assert (
        classify_tool_activity_family(tool_name="", provider_tool_type="mcp_call")
        == "remote_tool"
    )
    assert classify_tool_activity_family(tool_name="web_search") == "web_search"


def test_classify_tool_activity_family_normalizes_lookup_tokens() -> None:
    assert (
        classify_tool_activity_family(tool_name="ignored", provider_tool_type=" MCP_CALL ")
        == "remote_tool"
    )
    assert classify_tool_activity_family(tool_name=" WEB_SEARCH ") == "web_search"
    assert (
        classify_tool_activity_family(tool_name=" SERVER/MCP_LIST_TOOLS ")
        == "remote_tool_listing"
    )
