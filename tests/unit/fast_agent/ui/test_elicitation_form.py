from typing import TYPE_CHECKING

import pytest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.validation import ValidationError
from prompt_toolkit.widgets import Checkbox, RadioList

from fast_agent.human_input.form_elements import ValidatedCheckboxList
from fast_agent.ui.elicitation_form import ElicitationForm, FormatValidator, SimpleStringValidator

if TYPE_CHECKING:
    from mcp.types import ElicitRequestedSchema


def test_elicitation_form_creates_widgets_for_common_field_types() -> None:
    schema: "ElicitRequestedSchema" = {
        "type": "object",
        "properties": {
            "email": {"type": "string", "format": "email"},
            "enabled": {"type": "boolean", "default": True},
            "priority": {
                "type": "string",
                "enum": ["high", "low"],
                "enumNames": ["High", "Low"],
            },
            "tags": {
                "type": "array",
                "items": {
                    "enum": ["docs", "tests"],
                    "enumNames": ["Docs", "Tests"],
                },
                "default": ["docs"],
                "minItems": 1,
            },
        },
        "required": ["email"],
    }

    form = ElicitationForm(schema, "Please fill the fields", "planner", "server-a")

    email_widget = form.field_widgets["email"]
    enabled_widget = form.field_widgets["enabled"]
    priority_widget = form.field_widgets["priority"]
    tags_widget = form.field_widgets["tags"]

    assert isinstance(email_widget, Buffer)
    assert isinstance(email_widget.validator, FormatValidator)
    assert isinstance(enabled_widget, Checkbox)
    assert enabled_widget.checked is True
    assert isinstance(priority_widget, RadioList)
    assert isinstance(tags_widget, ValidatedCheckboxList)
    assert list(tags_widget.current_values) == ["docs"]


def test_elicitation_form_uses_radio_list_for_anyof_string_enums() -> None:
    schema: "ElicitRequestedSchema" = {
        "type": "object",
        "properties": {
            "choice": {
                "type": "string",
                "anyOf": [
                    {"enum": ["alpha", "beta"], "enumNames": ["Alpha", "Beta"]},
                    {"type": "null"},
                ],
                "default": "beta",
            },
        },
    }

    form = ElicitationForm(schema, "Choose one", "planner", "server-a")

    choice_widget = form.field_widgets["choice"]
    assert isinstance(choice_widget, RadioList)
    assert choice_widget.values == [("alpha", "Alpha"), ("beta", "Beta")]
    assert choice_widget.current_value == "beta"


def test_elicitation_form_tolerates_short_enum_names() -> None:
    schema: "ElicitRequestedSchema" = {
        "type": "object",
        "properties": {
            "choice": {
                "type": "string",
                "enum": ["alpha", "beta"],
                "enumNames": ["Alpha"],
            },
        },
    }

    form = ElicitationForm(schema, "Choose one", "planner", "server-a")

    choice_widget = form.field_widgets["choice"]
    assert isinstance(choice_widget, RadioList)
    assert choice_widget.values == [("alpha", "Alpha"), ("beta", "beta")]


def test_elicitation_form_navigation_mode_is_instance_scoped() -> None:
    schema: "ElicitRequestedSchema" = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
    }
    first = ElicitationForm(schema, "First", "planner", "server-a")
    second = ElicitationForm(schema, "Second", "planner", "server-a")

    first._toggle_text_navigation_mode()

    assert first._field_navigation_mode() is False
    assert second._field_navigation_mode() is True
    assert "TEXT MODE" in str(first._toolbar_text())
    assert "FIELD MODE" in str(second._toolbar_text())


def test_elicitation_form_toolbar_hidden_is_instance_state() -> None:
    schema: "ElicitRequestedSchema" = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
    }
    form = ElicitationForm(schema, "Need input", "planner", "server-a")

    assert form._toolbar_hidden is False
    form._toolbar_hidden = True

    assert str(form._toolbar_text()) == "FormattedText([])"


@pytest.mark.parametrize(
    ("validator", "value", "message"),
    [
        (SimpleStringValidator(min_length=5), "four", "Need 1 more char"),
        (SimpleStringValidator(min_length=6), "four", "Need 2 more chars"),
        (SimpleStringValidator(max_length=4), "12345", "Too long by 1 char"),
        (SimpleStringValidator(max_length=4), "123456", "Too long by 2 chars"),
    ],
)
def test_string_validator_formats_length_messages(
    validator: SimpleStringValidator,
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        validator.validate(Document(value))

    assert exc_info.value.message == message


@pytest.mark.parametrize(
    ("format_type", "invalid_value", "message"),
    [
        ("email", "not-an-email", "Invalid email format"),
        ("uri", "not a uri", "Invalid URI format"),
        ("date", "2026-99-99", "Invalid date (use YYYY-MM-DD)"),
        ("date-time", "tomorrow", "Invalid datetime (use ISO 8601)"),
    ],
)
def test_format_validator_reports_format_specific_messages(
    format_type: str,
    invalid_value: str,
    message: str,
) -> None:
    validator = FormatValidator(format_type)

    with pytest.raises(ValidationError) as exc_info:
        validator.validate(Document(invalid_value))
    assert exc_info.value.message == message


@pytest.mark.asyncio
async def test_elicitation_form_validates_and_collects_typed_data() -> None:
    schema: "ElicitRequestedSchema" = {
        "type": "object",
        "properties": {
            "count": {"type": "integer"},
            "ratio": {"type": "number"},
            "notes": {"type": "string"},
            "enabled": {"type": "boolean", "default": False},
            "choice": {"type": "string", "enum": ["a", "b"]},
            "tags": {
                "type": "array",
                "items": {"enum": ["x", "y"]},
                "minItems": 1,
            },
        },
        "required": ["count", "choice", "tags"],
    }

    form = ElicitationForm(schema, "Collect values", "planner", "server-a")

    count_widget = form.field_widgets["count"]
    ratio_widget = form.field_widgets["ratio"]
    notes_widget = form.field_widgets["notes"]
    enabled_widget = form.field_widgets["enabled"]
    choice_widget = form.field_widgets["choice"]
    tags_widget = form.field_widgets["tags"]

    assert isinstance(count_widget, Buffer)
    assert isinstance(ratio_widget, Buffer)
    assert isinstance(notes_widget, Buffer)
    assert isinstance(enabled_widget, Checkbox)
    assert isinstance(choice_widget, RadioList)
    assert isinstance(tags_widget, ValidatedCheckboxList)

    count_widget.text = "7"
    ratio_widget.text = "3.5"
    notes_widget.text = "hello"
    enabled_widget.checked = True
    choice_widget.current_value = "b"
    tags_widget.current_values = ["x"]

    validation = form._validate_form()

    assert validation.is_valid is True
    assert validation.error_message is None
    assert form._get_form_data() == {
        "count": 7,
        "ratio": 3.5,
        "notes": "hello",
        "enabled": True,
        "choice": "b",
        "tags": ["x"],
    }


def test_elicitation_form_reports_missing_required_fields() -> None:
    schema: "ElicitRequestedSchema" = {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "title": "Summary"},
            "choice": {"type": "string", "enum": ["yes", "no"], "title": "Choice"},
        },
        "required": ["summary", "choice"],
    }

    form = ElicitationForm(schema, "Need input", "planner", "server-a")

    validation = form._validate_form()

    assert validation.is_valid is False
    assert validation.error_message == "'Summary' is required"


@pytest.mark.asyncio
async def test_elicitation_form_reports_widget_validation_errors() -> None:
    schema: "ElicitRequestedSchema" = {
        "type": "object",
        "properties": {
            "email": {"type": "string", "format": "email", "title": "Email"},
        },
        "required": ["email"],
    }

    form = ElicitationForm(schema, "Need email", "planner", "server-a")
    email_widget = form.field_widgets["email"]
    assert isinstance(email_widget, Buffer)

    email_widget.text = "not-an-email"
    email_widget.validate(set_cursor=False)
    validation = form._validate_form()

    assert validation.is_valid is False
    assert validation.error_message == "'Email': Invalid email format"


def test_elicitation_form_formats_singular_string_length_hints() -> None:
    schema: "ElicitRequestedSchema" = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "title": "Summary",
                "minLength": 1,
                "maxLength": 1,
            },
        },
    }

    form = ElicitationForm(schema, "Need input", "planner", "server-a")

    assert form._string_field_hints(schema["properties"]["summary"]) == ["min 1 char", "max 1 char"]
