from fast_agent.ui.attachment_indicator import DraftAttachmentSummary
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


def test_build_middle_segment_renders_attachment_indicator() -> None:
    middle = _build_middle_segment(
        ToolbarAgentState(
            model_display="gpt-4.1",
            model_name="gpt-4.1",
            model_gauges="RG",
            tdv_segment="TVD",
            service_tier_indicator="FAST",
            web_search_indicator="WEB",
            turn_count=3,
        ),
        shortcut_text="",
        attachment_summary=DraftAttachmentSummary(
            count=2,
            mime_types=("image/png",),
            any_questionable=False,
        ),
    )

    assert "▲2" in middle
    assert middle.index("TVD") < middle.index("▲2") < middle.index("RG") < middle.index("gpt-4.1")
    assert middle.index("gpt-4.1") < middle.index("FAST") < middle.index("WEB")
