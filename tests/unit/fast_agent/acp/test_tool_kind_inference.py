from __future__ import annotations

from typing import get_args

import pytest
from acp.schema import ToolKind

from fast_agent.acp.tool_kind_inference import TOOL_KIND_KEYWORD_GROUPS, infer_tool_kind


def test_tool_kind_keyword_groups_use_valid_acp_tool_kinds() -> None:
    valid_kinds = set(get_args(ToolKind))

    assert {group.kind for group in TOOL_KIND_KEYWORD_GROUPS} <= valid_kinds


def test_tool_kind_keywords_are_unique_across_ordered_groups() -> None:
    keyword_owners: dict[str, str] = {}

    for group in TOOL_KIND_KEYWORD_GROUPS:
        for keyword in group.keywords:
            previous_owner = keyword_owners.setdefault(keyword, group.kind)
            assert previous_owner == group.kind, (
                f"keyword {keyword!r} is shared by {previous_owner!r} and {group.kind!r}"
            )


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        ("read_file", "read"),
        ("write_file", "edit"),
        ("delete_file", "delete"),
        ("move_file", "move"),
        ("grep_code", "search"),
        ("run_shell", "execute"),
        ("plan_next_step", "think"),
        ("http_request", "fetch"),
        ("fetch_url", "fetch"),
        ("READ_FILE", "read"),
        ("Delete_Item", "delete"),
        ("EXECUTE_CMD", "execute"),
        ("custom_tool", "other"),
        ("process_data", "other"),
    ],
)
def test_infer_tool_kind_uses_ordered_keyword_groups(
    tool_name: str, expected: "ToolKind"
) -> None:
    assert infer_tool_kind(tool_name) == expected


def test_infer_tool_kind_treats_fetch_as_network_access() -> None:
    assert infer_tool_kind("fetch_record") == "fetch"


@pytest.mark.parametrize(
    ("tool_name", "expected"),
    [
        ("create_file", "edit"),
        ("copy_document", "move"),
        ("locate_symbol", "search"),
        ("analyze_plan", "think"),
        ("curl_endpoint", "fetch"),
    ],
)
def test_shared_tool_kind_inference_covers_execution_keywords(
    tool_name: str,
    expected: "ToolKind",
) -> None:
    assert infer_tool_kind(tool_name) == expected
