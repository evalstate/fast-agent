"""Custom form elements for elicitation forms."""

from collections.abc import Sequence
from typing import TypeVar

from prompt_toolkit.formatted_text import AnyFormattedText
from prompt_toolkit.validation import ValidationError
from prompt_toolkit.widgets import CheckboxList

_T = TypeVar("_T")


class ValidatedCheckboxList(CheckboxList[_T]):
    """CheckboxList with min/max items validation."""

    def __init__(
        self,
        values: Sequence[tuple[_T, AnyFormattedText]],
        default_values: Sequence[_T] | None = None,
        min_items: int | None = None,
        max_items: int | None = None,
    ):
        """
        Initialize checkbox list with validation.

        Args:
            values: List of (value, label) tuples
            default_values: Initially selected values
            min_items: Minimum number of items that must be selected
            max_items: Maximum number of items that can be selected
        """
        super().__init__(values, default_values=default_values)
        self.min_items = min_items
        self.max_items = max_items

    @property
    def validation_error(self) -> ValidationError | None:
        """
        Check if current selection is valid.

        Returns:
            ValidationError if invalid, None if valid
        """
        selected_count = len(self.current_values)

        if self.min_items is not None and selected_count < self.min_items:
            if self.min_items == 1:
                message = "At least 1 selection required"
            else:
                message = f"At least {self.min_items} selections required"
            return ValidationError(message=message)

        if self.max_items is not None and selected_count > self.max_items:
            if self.max_items == 1:
                message = "Only 1 selection allowed"
            else:
                message = f"Maximum {self.max_items} selections allowed"
            return ValidationError(message=message)

        return None
