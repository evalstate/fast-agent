import json

from fast_agent.batch.template import DEFAULT_ROW_TEMPLATE, render_row_template


def test_default_template_dumps_pretty_row_json():
    rendered = render_row_template(DEFAULT_ROW_TEMPLATE, {"id": "1", "count": 2})

    assert rendered.error is None
    assert rendered.text is not None
    assert "Input record:" in rendered.text
    assert json.dumps({"id": "1", "count": 2}, indent=2) in rendered.text


def test_template_renders_field_placeholders_and_row_json():
    rendered = render_row_template(
        "Message: {{message}}\nPayload:\n{{row_json}}",
        {"message": "hello", "tags": ["a"]},
    )

    assert rendered.error is None
    assert rendered.text is not None
    assert "Message: hello" in rendered.text
    assert '"tags": [\n    "a"\n  ]' in rendered.text


def test_missing_template_field_returns_row_error():
    rendered = render_row_template("{{missing}}", {"message": "hello"})

    assert rendered.text is None
    assert rendered.error is not None
    assert rendered.error.type == "MissingTemplateField"
    assert rendered.error.message == "Missing template field: missing"


def test_missing_template_fields_returns_plural_row_error():
    rendered = render_row_template("{{first}} {{second}}", {"message": "hello"})

    assert rendered.text is None
    assert rendered.error is not None
    assert rendered.error.type == "MissingTemplateField"
    assert rendered.error.message == "Missing template fields: first, second"
