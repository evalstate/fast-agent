from __future__ import annotations

from fast_agent.core.template_render import extract_template_variables, render_template_text


def test_extract_template_variables_normalizes_placeholder_names() -> None:
    assert extract_template_variables("{{ name }} {{   }} {{ topic }}") == {"name", "topic"}


def test_render_template_text_preserves_blank_and_missing_placeholders() -> None:
    result = render_template_text(
        "{{ name }} {{ missing }} {{   }}",
        {"name": "Ada"},
    )

    assert result.text == "Ada {{ missing }} {{   }}"
    assert result.missing == ("missing",)
