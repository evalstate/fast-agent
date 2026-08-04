from fast_agent.ui.apply_patch_preview import style_apply_patch_preview_text
from fast_agent.ui.edit_file_preview import build_edit_file_preview, format_edit_file_preview


def test_edit_file_preview_uses_unified_diff_and_shared_colours() -> None:
    preview = build_edit_file_preview(
        {
            "path": "src/example.py",
            "old_string": "old = 1\n",
            "new_string": "new = 2\n",
        }
    )

    assert preview is not None
    text = format_edit_file_preview(preview)
    assert text == (
        "edit_file preview: src/example.py\n"
        "--- src/example.py\n"
        "+++ src/example.py\n"
        "@@ -1 +1 @@\n"
        "-old = 1\n"
        "+new = 2"
    )

    span_styles = {str(span.style) for span in style_apply_patch_preview_text(text).spans}
    assert "red" in span_styles
    assert "green" in span_styles
