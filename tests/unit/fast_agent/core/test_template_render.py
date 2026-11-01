from __future__ import annotations

import json

import pytest

from fast_agent.core.agent_tool_shape import render_agent_tool_arguments
from fast_agent.core.template_render import (
    extract_template_variables,
    render_mapping_template,
    render_template_text,
)


def test_extract_template_variables_normalizes_placeholder_names() -> None:
    assert extract_template_variables("{{ name }} {{   }} {{ topic }}") == {"name", "topic"}


def test_render_template_text_preserves_blank_and_missing_placeholders() -> None:
    result = render_template_text(
        "{{ name }} {{ missing }} {{   }}",
        {"name": "Ada"},
    )

    assert result.text == "Ada {{ missing }} {{   }}"
    assert result.missing == ("missing",)


@pytest.mark.parametrize("placeholder", ["row_json", "arguments_json"])
@pytest.mark.parametrize("value", ["a single field", None, {"nested": True}])
def test_full_mapping_placeholder_cannot_be_shadowed_by_field(placeholder, value):
    values = {placeholder: value, "topic": "billing"}

    result = render_mapping_template(
        "{{" + placeholder + "}}", values, json_placeholder=placeholder
    )

    assert json.loads(result.text) == values
    assert result.missing == ()


def test_agent_tool_template_preserves_all_arguments_when_name_collides():
    arguments = {"arguments_json": "source payload", "topic": "billing"}

    rendered = render_agent_tool_arguments(arguments, render_arguments="{{arguments_json}}")

    assert json.loads(rendered) == arguments
