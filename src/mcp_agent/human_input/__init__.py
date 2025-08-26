"""Human input modules for forms and elicitation."""

# Export the simple form API
# Export field types and schema builder
from mcp_agent.human_input.form_fields import (
    BooleanField,
    EnumField,
    FormSchema,
    IntegerField,
    NumberField,
    # Field classes
    StringField,
    boolean,
    choice,
    date,
    datetime,
    email,
    integer,
    number,
    # Convenience functions
    string,
    url,
)
from mcp_agent.human_input.simple_form import ask, ask_sync, form, form_sync

__all__ = [
    # Form functions
    "form",
    "form_sync",
    "ask",
    "ask_sync",
    # Schema builder
    "FormSchema",
    # Field classes
    "StringField",
    "IntegerField",
    "NumberField",
    "BooleanField",
    "EnumField",
    # Field convenience functions
    "string",
    "email",
    "url",
    "date",
    "datetime",
    "integer",
    "number",
    "boolean",
    "choice",
]
