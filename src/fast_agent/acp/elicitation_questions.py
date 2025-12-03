"""
Convert JSON Schema to friendly Q&A questions for interactive elicitation.

This module transforms JSON Schema field definitions into conversational questions
that can be asked one at a time through ACP's prompt interface.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ElicitationQuestion:
    """Represents a single question to ask the user."""

    field_name: str
    """The JSON Schema property name."""

    question_text: str
    """The friendly question to display to the user."""

    field_type: str
    """The JSON Schema type: string, number, integer, boolean, enum."""

    required: bool
    """Whether this field is required."""

    default: Any = None
    """Default value if user skips (optional fields only)."""

    # Validation constraints
    min_value: float | None = None
    max_value: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    format_type: str | None = None  # email, uri, date, date-time

    # For enum/radio fields
    options: list[tuple[str, str]] = field(default_factory=list)
    """List of (value, display_label) tuples for enum fields."""

    hint_text: str | None = None
    """Additional hint to show after the question."""


def _format_validation_hints(question: ElicitationQuestion) -> str:
    """Generate user-friendly validation hints."""
    hints = []

    if question.format_type:
        format_examples = {
            "email": "e.g., user@example.com",
            "uri": "e.g., https://example.com",
            "date": "format: YYYY-MM-DD",
            "date-time": "format: YYYY-MM-DDTHH:MM:SS",
        }
        if question.format_type in format_examples:
            hints.append(format_examples[question.format_type])

    if question.min_length is not None and question.max_length is not None:
        hints.append(f"{question.min_length}-{question.max_length} characters")
    elif question.min_length is not None:
        hints.append(f"at least {question.min_length} characters")
    elif question.max_length is not None:
        hints.append(f"up to {question.max_length} characters")

    if question.min_value is not None and question.max_value is not None:
        hints.append(f"between {question.min_value} and {question.max_value}")
    elif question.min_value is not None:
        hints.append(f"minimum: {question.min_value}")
    elif question.max_value is not None:
        hints.append(f"maximum: {question.max_value}")

    if question.pattern:
        hints.append(f"pattern: {question.pattern}")

    return ", ".join(hints) if hints else ""


def _extract_string_constraints(field_def: dict[str, Any]) -> dict[str, Any]:
    """Extract string constraints from field definition, handling anyOf schemas."""
    constraints: dict[str, Any] = {}

    # Check direct constraints
    if field_def.get("minLength") is not None:
        constraints["minLength"] = field_def["minLength"]
    if field_def.get("maxLength") is not None:
        constraints["maxLength"] = field_def["maxLength"]
    if field_def.get("pattern") is not None:
        constraints["pattern"] = field_def["pattern"]
    if field_def.get("format") is not None:
        constraints["format"] = field_def["format"]

    # Check anyOf constraints (for Optional fields)
    if "anyOf" in field_def:
        for variant in field_def["anyOf"]:
            if variant.get("type") == "string":
                if variant.get("minLength") is not None:
                    constraints["minLength"] = variant["minLength"]
                if variant.get("maxLength") is not None:
                    constraints["maxLength"] = variant["maxLength"]
                if variant.get("pattern") is not None:
                    constraints["pattern"] = variant["pattern"]
                if variant.get("format") is not None:
                    constraints["format"] = variant["format"]
                break

    return constraints


def schema_to_questions(schema: dict[str, Any]) -> list[ElicitationQuestion]:
    """
    Convert a JSON Schema to a list of friendly questions.

    Args:
        schema: JSON Schema with properties, required fields, etc.

    Returns:
        List of ElicitationQuestion objects in display order.
    """
    questions: list[ElicitationQuestion] = []

    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))

    for field_name, field_def in properties.items():
        is_required = field_name in required_fields

        # Extract basic info
        title = field_def.get("title", _humanize_field_name(field_name))
        description = field_def.get("description", "")
        default_value = field_def.get("default")
        field_type = field_def.get("type", "string")

        # Build the question text
        question_text = title
        if description:
            question_text = f"{title}\n   {description}"

        # Handle different field types
        if field_type == "boolean":
            question = ElicitationQuestion(
                field_name=field_name,
                question_text=question_text,
                field_type="boolean",
                required=is_required,
                default=default_value,
                hint_text="Answer: yes/no (or y/n, true/false)",
            )

        elif field_type == "string" and "enum" in field_def:
            # Enum/radio field
            enum_values = field_def["enum"]
            enum_names = field_def.get("enumNames", enum_values)
            options = list(zip(enum_values, enum_names))

            # Build options display
            options_text = "\n".join(
                [f"   {i + 1}. {name}" for i, (_, name) in enumerate(options)]
            )

            question = ElicitationQuestion(
                field_name=field_name,
                question_text=f"{question_text}\n{options_text}",
                field_type="enum",
                required=is_required,
                default=default_value,
                options=options,
                hint_text="Enter the number or value of your choice",
            )

        elif field_type in ["number", "integer"]:
            question = ElicitationQuestion(
                field_name=field_name,
                question_text=question_text,
                field_type=field_type,
                required=is_required,
                default=default_value,
                min_value=field_def.get("minimum"),
                max_value=field_def.get("maximum"),
            )

        else:
            # String field (default)
            constraints = _extract_string_constraints(field_def)

            question = ElicitationQuestion(
                field_name=field_name,
                question_text=question_text,
                field_type="string",
                required=is_required,
                default=default_value,
                min_length=constraints.get("minLength"),
                max_length=constraints.get("maxLength"),
                pattern=constraints.get("pattern"),
                format_type=constraints.get("format"),
            )

        # Generate hint text from validation constraints
        validation_hints = _format_validation_hints(question)
        if validation_hints and not question.hint_text:
            question.hint_text = validation_hints
        elif validation_hints and question.hint_text:
            question.hint_text = f"{question.hint_text} ({validation_hints})"

        questions.append(question)

    return questions


def _humanize_field_name(field_name: str) -> str:
    """Convert snake_case or camelCase to human-readable title."""
    # Handle snake_case
    result = field_name.replace("_", " ")

    # Handle camelCase - insert space before capital letters
    humanized = ""
    for i, char in enumerate(result):
        if char.isupper() and i > 0 and not result[i - 1].isupper():
            humanized += " "
        humanized += char

    return humanized.title()


def format_question_for_display(
    question: ElicitationQuestion,
    question_number: int,
    total_questions: int,
) -> str:
    """
    Format a question for display to the user via ACP.

    Returns a nicely formatted string with question number, text, and hints.
    """
    lines = []

    # Question header with progress
    required_marker = " *" if question.required else ""
    lines.append(f"**Question {question_number} of {total_questions}**{required_marker}")
    lines.append("")

    # Question text
    lines.append(question.question_text)

    # Default value hint
    if question.default is not None:
        lines.append("")
        lines.append(f"Default: {question.default} (press Enter to use default)")

    # Validation hints
    if question.hint_text:
        lines.append("")
        lines.append(f"_{question.hint_text}_")

    # Optional field hint
    if not question.required and question.default is None:
        lines.append("")
        lines.append("_This field is optional. Type 'skip' to skip._")

    return "\n".join(lines)


def format_elicitation_intro(
    schema: dict[str, Any],
    message: str,
    agent_name: str,
    server_name: str,
    total_questions: int,
) -> str:
    """
    Format the introduction message for an elicitation session.

    Returns a friendly intro explaining what's about to happen.
    """
    # Get title from schema or use generic
    title = schema.get("title", "Information Request")

    lines = [
        "---",
        f"**{title}**",
        "",
        f"_From: {server_name} via {agent_name}_",
        "",
        message,
        "",
        "---",
        "",
        f"I'll ask you {total_questions} question{'s' if total_questions != 1 else ''} to complete this request.",
        "",
        "**Commands you can use:**",
        "- Type your answer and press Enter",
        "- `skip` - Skip an optional field",
        "- `back` - Go back to the previous question",
        "- `cancel` - Cancel this request",
        "- `decline` - Decline to provide information",
        "",
        "---",
        "",
    ]

    return "\n".join(lines)


def format_validation_error(error_message: str) -> str:
    """Format a validation error message."""
    return f"**Validation Error:** {error_message}\n\nPlease try again:"


def format_completion_summary(
    schema: dict[str, Any],
    collected_data: dict[str, Any],
) -> str:
    """
    Format a summary of collected data before submission.

    Returns a formatted summary for user confirmation.
    """
    title = schema.get("title", "Information Request")
    properties = schema.get("properties", {})

    lines = [
        "---",
        f"**{title} - Summary**",
        "",
    ]

    for field_name, value in collected_data.items():
        field_def = properties.get(field_name, {})
        display_name = field_def.get("title", _humanize_field_name(field_name))

        # Format the value for display
        if isinstance(value, bool):
            display_value = "Yes" if value else "No"
        elif value is None:
            display_value = "_skipped_"
        else:
            display_value = str(value)

        lines.append(f"- **{display_name}:** {display_value}")

    lines.extend([
        "",
        "---",
        "",
        "Type `submit` to confirm, `back` to edit, or `cancel` to abort.",
    ])

    return "\n".join(lines)
