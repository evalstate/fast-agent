"""High-level field types for elicitation forms with default support."""

from dataclasses import dataclass
from typing import Any, Union


@dataclass
class StringField:
    """String field with validation and default support."""

    title: str | None = None
    description: str | None = None
    default: str | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    format: str | None = None  # email, uri, date, date-time

    def to_schema(self) -> dict[str, Any]:
        """Convert to MCP elicitation schema format."""
        schema: dict[str, Any] = {"type": "string"}

        if self.title:
            schema["title"] = self.title
        if self.description:
            schema["description"] = self.description
        if self.default is not None:
            schema["default"] = self.default
        if self.min_length is not None:
            schema["minLength"] = self.min_length
        if self.max_length is not None:
            schema["maxLength"] = self.max_length
        if self.pattern is not None:
            schema["pattern"] = self.pattern
        if self.format:
            schema["format"] = self.format

        return schema


@dataclass
class IntegerField:
    """Integer field with validation and default support."""

    title: str | None = None
    description: str | None = None
    default: int | None = None
    minimum: int | None = None
    maximum: int | None = None

    def to_schema(self) -> dict[str, Any]:
        """Convert to MCP elicitation schema format."""
        schema: dict[str, Any] = {"type": "integer"}

        if self.title:
            schema["title"] = self.title
        if self.description:
            schema["description"] = self.description
        if self.default is not None:
            schema["default"] = self.default
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum

        return schema


@dataclass
class NumberField:
    """Number (float) field with validation and default support."""

    title: str | None = None
    description: str | None = None
    default: float | None = None
    minimum: float | None = None
    maximum: float | None = None

    def to_schema(self) -> dict[str, Any]:
        """Convert to MCP elicitation schema format."""
        schema: dict[str, Any] = {"type": "number"}

        if self.title:
            schema["title"] = self.title
        if self.description:
            schema["description"] = self.description
        if self.default is not None:
            schema["default"] = self.default
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum

        return schema


@dataclass
class BooleanField:
    """Boolean field with default support."""

    title: str | None = None
    description: str | None = None
    default: bool | None = None

    def to_schema(self) -> dict[str, Any]:
        """Convert to MCP elicitation schema format."""
        schema: dict[str, Any] = {"type": "boolean"}

        if self.title:
            schema["title"] = self.title
        if self.description:
            schema["description"] = self.description
        if self.default is not None:
            schema["default"] = self.default

        return schema


@dataclass
class EnumField:
    """Enum/choice field with default support."""

    choices: list[str]
    choice_names: list[str] | None = None  # Human-readable names
    title: str | None = None
    description: str | None = None
    default: str | None = None

    def to_schema(self) -> dict[str, Any]:
        """Convert to MCP elicitation schema format."""
        schema: dict[str, Any] = {"type": "string", "enum": self.choices}

        if self.title:
            schema["title"] = self.title
        if self.description:
            schema["description"] = self.description
        if self.default is not None:
            schema["default"] = self.default
        if self.choice_names:
            schema["enumNames"] = self.choice_names

        return schema


# Field type union
FieldType = Union[StringField, IntegerField, NumberField, BooleanField, EnumField]


class FormSchema:
    """High-level form schema builder."""

    def __init__(self, **fields: FieldType):
        """Create a form schema with named fields."""
        self.fields = fields
        self._required_fields: list[str] = []

    def required(self, *field_names: str) -> "FormSchema":
        """Mark fields as required."""
        self._required_fields.extend(field_names)
        return self

    def to_schema(self) -> dict[str, Any]:
        """Convert to MCP ElicitRequestedSchema format."""
        properties = {}

        for field_name, field in self.fields.items():
            properties[field_name] = field.to_schema()

        schema: dict[str, Any] = {"type": "object", "properties": properties}

        if self._required_fields:
            schema["required"] = self._required_fields

        return schema


# Convenience functions for creating fields
def string(
    title: str | None = None,
    description: str | None = None,
    default: str | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    pattern: str | None = None,
    format: str | None = None,
) -> StringField:
    """Create a string field."""
    return StringField(title, description, default, min_length, max_length, pattern, format)


def email(
    title: str | None = None, description: str | None = None, default: str | None = None
) -> StringField:
    """Create an email field."""
    return StringField(title, description, default, format="email")


def url(
    title: str | None = None, description: str | None = None, default: str | None = None
) -> StringField:
    """Create a URL field."""
    return StringField(title, description, default, format="uri")


def date(
    title: str | None = None, description: str | None = None, default: str | None = None
) -> StringField:
    """Create a date field."""
    return StringField(title, description, default, format="date")


def datetime(
    title: str | None = None, description: str | None = None, default: str | None = None
) -> StringField:
    """Create a datetime field."""
    return StringField(title, description, default, format="date-time")


def integer(
    title: str | None = None,
    description: str | None = None,
    default: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> IntegerField:
    """Create an integer field."""
    return IntegerField(title, description, default, minimum, maximum)


def number(
    title: str | None = None,
    description: str | None = None,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> NumberField:
    """Create a number field."""
    return NumberField(title, description, default, minimum, maximum)


def boolean(
    title: str | None = None, description: str | None = None, default: bool | None = None
) -> BooleanField:
    """Create a boolean field."""
    return BooleanField(title, description, default)


def choice(
    choices: list[str],
    choice_names: list[str] | None = None,
    title: str | None = None,
    description: str | None = None,
    default: str | None = None,
) -> EnumField:
    """Create a choice/enum field."""
    return EnumField(choices, choice_names, title, description, default)
