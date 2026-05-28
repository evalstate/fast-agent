import json

from fast_agent.batch.template import DEFAULT_ROW_TEMPLATE, render_row_template


def test_default_template_dumps_pretty_row_json():
    rendered, error = render_row_template(DEFAULT_ROW_TEMPLATE, {"id": "1", "count": 2})

    assert error is None
    assert rendered is not None
    assert "Input record:" in rendered
    assert json.dumps({"id": "1", "count": 2}, indent=2) in rendered


def test_template_renders_field_placeholders_and_row_json():
    rendered, error = render_row_template(
        "Message: {{message}}\nPayload:\n{{row_json}}",
        {"message": "hello", "tags": ["a"]},
    )

    assert error is None
    assert rendered is not None
    assert "Message: hello" in rendered
    assert '"tags": [\n    "a"\n  ]' in rendered


def test_missing_template_field_returns_row_error():
    rendered, error = render_row_template("{{missing}}", {"message": "hello"})

    assert rendered is None
    assert error is not None
    assert error.type == "MissingTemplateField"

