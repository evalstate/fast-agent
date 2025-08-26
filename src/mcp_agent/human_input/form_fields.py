"""High-level field types for elicitation forms with default support."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class StringField:
    """String field with validation and default support."""

    title: Optional[str] = None
    description: Optional[str] = None
    default: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    format: Optional[str] = None  # email, uri, date, date-time

    def to_schema(self) -> Dict[str, Any]:
        """Convert to MCP elicitation schema format."""
        schema: Dict[str, Any] = {"type": "string"}

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
        if self.format:
            schema["format"] = self.format

        return schema


@dataclass
class IntegerField:
    """Integer field with validation and default support."""

    title: Optional[str] = None
    description: Optional[str] = None
    default: Optional[int] = None
    minimum: Optional[int] = None
    maximum: Optional[int] = None

    def to_schema(self) -> Dict[str, Any]:
        """Convert to MCP elicitation schema format."""
        schema: Dict[str, Any] = {"type": "integer"}

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

    title: Optional[str] = None
    description: Optional[str] = None
    default: Optional[float] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def to_schema(self) -> Dict[str, Any]:
        """Convert to MCP elicitation schema format."""
        schema: Dict[str, Any] = {"type": "number"}

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

    title: Optional[str] = None
    description: Optional[str] = None
    default: Optional[bool] = None

    def to_schema(self) -> Dict[str, Any]:
        """Convert to MCP elicitation schema format."""
        schema: Dict[str, Any] = {"type": "boolean"}

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

    choices: List[str]
    choice_names: Optional[List[str]] = None  # Human-readable names
    title: Optional[str] = None
    description: Optional[str] = None
    default: Optional[str] = None

    def to_schema(self) -> Dict[str, Any]:
        """Convert to MCP elicitation schema format."""
        schema: Dict[str, Any] = {"type": "string", "enum": self.choices}

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
        self._required_fields: List[str] = []

    def required(self, *field_names: str) -> "FormSchema":
        """Mark fields as required."""
        self._required_fields.extend(field_names)
        return self

    def to_schema(self) -> Dict[str, Any]:
        """Convert to MCP ElicitRequestedSchema format."""
        properties = {}

        for field_name, field in self.fields.items():
            properties[field_name] = field.to_schema()

        schema: Dict[str, Any] = {"type": "object", "properties": properties}

        if self._required_fields:
            schema["required"] = self._required_fields

        return schema


# Convenience functions for creating fields
def string(
    title: Optional[str] = None,
    description: Optional[str] = None,
    default: Optional[str] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    format: Optional[str] = None,
) -> StringField:
    """Create a string field."""
    return StringField(title, description, default, min_length, max_length, format)


def email(
    title: Optional[str] = None, description: Optional[str] = None, default: Optional[str] = None
) -> StringField:
    """Create an email field."""
    return StringField(title, description, default, format="email")


def url(
    title: Optional[str] = None, description: Optional[str] = None, default: Optional[str] = None
) -> StringField:
    """Create a URL field."""
    return StringField(title, description, default, format="uri")


def date(
    title: Optional[str] = None, description: Optional[str] = None, default: Optional[str] = None
) -> StringField:
    """Create a date field."""
    return StringField(title, description, default, format="date")


def datetime(
    title: Optional[str] = None, description: Optional[str] = None, default: Optional[str] = None
) -> StringField:
    """Create a datetime field."""
    return StringField(title, description, default, format="date-time")


def integer(
    title: Optional[str] = None,
    description: Optional[str] = None,
    default: Optional[int] = None,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> IntegerField:
    """Create an integer field."""
    return IntegerField(title, description, default, minimum, maximum)


def number(
    title: Optional[str] = None,
    description: Optional[str] = None,
    default: Optional[float] = None,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> NumberField:
    """Create a number field."""
    return NumberField(title, description, default, minimum, maximum)


def boolean(
    title: Optional[str] = None, description: Optional[str] = None, default: Optional[bool] = None
) -> BooleanField:
    """Create a boolean field."""
    return BooleanField(title, description, default)


def choice(
    choices: List[str],
    choice_names: Optional[List[str]] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    default: Optional[str] = None,
) -> EnumField:
    """Create a choice/enum field."""
    return EnumField(choices, choice_names, title, description, default)
