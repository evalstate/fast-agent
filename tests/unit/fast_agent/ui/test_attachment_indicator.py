from __future__ import annotations

from fast_agent.ui.attachment_indicator import (
    ATTACHMENT_GLYPH,
    ATTACHMENT_IDLE_COLOR,
    ATTACHMENT_QUESTIONABLE_COLOR,
    ATTACHMENT_SUPPORTED_COLOR,
    DraftAttachmentSummary,
    render_attachment_indicator,
    summarize_draft_attachments,
)


def test_summarize_draft_attachments_marks_supported_local_file(tmp_path) -> None:
    image = tmp_path / "image.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    summary = summarize_draft_attachments(
        f"describe ^file:{image}",
        model_name="gpt-4.1",
    )

    assert summary is not None
    assert summary.count == 1
    assert summary.any_questionable is False
    assert summary.mime_types == ("image/png",)


def test_summarize_draft_attachments_marks_missing_file_questionable() -> None:
    summary = summarize_draft_attachments(
        "describe ^file:/tmp/does-not-exist.png",
        model_name="gpt-4.1",
    )

    assert summary is not None
    assert summary.count == 1
    assert summary.any_questionable is True


def test_render_attachment_indicator_uses_red_count_for_questionable_summary() -> None:
    indicator = render_attachment_indicator(
        DraftAttachmentSummary(count=2, mime_types=("image/png",), any_questionable=True)
    )

    assert indicator == f"<style bg='{ATTACHMENT_QUESTIONABLE_COLOR}'>{ATTACHMENT_GLYPH}2</style>"


def test_render_attachment_indicator_formats_supported_indicator() -> None:
    indicator = render_attachment_indicator(
        DraftAttachmentSummary(count=1, mime_types=("image/png",), any_questionable=False)
    )

    assert indicator == f"<style bg='{ATTACHMENT_SUPPORTED_COLOR}'>{ATTACHMENT_GLYPH}1</style>"


def test_render_attachment_indicator_formats_idle_indicator() -> None:
    indicator = render_attachment_indicator(None)

    assert indicator == f"<style bg='{ATTACHMENT_IDLE_COLOR}'>{ATTACHMENT_GLYPH} </style>"
