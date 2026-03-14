from typing import TYPE_CHECKING

import pytest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.widgets import Checkbox, RadioList

from fast_agent.human_input.form_elements import ValidatedCheckboxList
from fast_agent.ui.elicitation_form import ElicitationForm, FormatValidator

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

    is_valid, message = form._validate_form()

    assert is_valid is True
    assert message is None
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

    is_valid, message = form._validate_form()

    assert is_valid is False
    assert message == "'Summary' is required"


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
    is_valid, message = form._validate_form()

    assert is_valid is False
    assert message == "'Email': Invalid email format"
