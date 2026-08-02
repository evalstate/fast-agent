from fast_agent.session import format_subagent_alias, subagent_alias_slug, subagent_task_preview


def test_subagent_alias_slug_prefers_label_and_is_path_safe() -> None:
    slug = subagent_alias_slug(
        label="  Investigate ../../ Item!  ",
        task="ignored",
    )

    assert format_subagent_alias(1, slug) == "01_investigate_item"


def test_subagent_alias_slug_uses_task_and_is_bounded() -> None:
    slug = subagent_alias_slug(
        label=None,
        task="Review the API contract and every possible edge case in detail",
    )

    assert slug == "review_the_api_contract_and"
    assert len(slug) <= 32


def test_subagent_alias_slug_has_safe_fallback() -> None:
    assert subagent_alias_slug(label=None, task="🦆 / ..") == "subagent"
    assert subagent_task_preview("  investigate\n\n  item  ") == "investigate item"
