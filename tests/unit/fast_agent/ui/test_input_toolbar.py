from fast_agent.ui.prompt.input_toolbar import ToolbarAgentState, _build_middle_segment


def test_build_middle_segment_prefixes_overlay_models() -> None:
    middle = _build_middle_segment(
        ToolbarAgentState(
            model_display="haikutiny",
            is_overlay_model=True,
            turn_count=3,
        ),
        shortcut_text="",
    )

    assert "▼haikutiny" in middle


def test_build_middle_segment_prefixes_codex_before_overlay() -> None:
    middle = _build_middle_segment(
        ToolbarAgentState(
            model_display="gpt-5-codex",
            is_codex_responses_model=True,
            is_overlay_model=True,
            turn_count=3,
        ),
        shortcut_text="",
    )

    assert "∞gpt-5-codex" in middle
    assert "▼gpt-5-codex" not in middle
