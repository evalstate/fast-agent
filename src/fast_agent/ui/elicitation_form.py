"""Simplified, robust elicitation form dialog."""

import re
from datetime import date, datetime
from typing import Any, Tuple

from mcp.types import ElicitRequestedSchema
from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.bindings.focus import focus_next, focus_previous
from prompt_toolkit.layout import HSplit, Layout, ScrollablePane, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.validation import ValidationError, Validator
from prompt_toolkit.widgets import (
    Button,
    Checkbox,
    Frame,
    Label,
    RadioList,
)
from pydantic import AnyUrl, EmailStr, TypeAdapter
from pydantic import ValidationError as PydanticValidationError

from fast_agent.human_input.form_elements import ValidatedCheckboxList
from fast_agent.ui.elicitation_style import ELICITATION_STYLE
from fast_agent.utils.async_utils import suppress_known_runtime_warnings

text_navigation_mode = False


class SimpleNumberValidator(Validator):
    """Simple number validator with real-time feedback."""

    def __init__(self, field_type: str, minimum: float | None = None, maximum: float | None = None):
        self.field_type = field_type
        self.minimum = minimum
        self.maximum = maximum

    def validate(self, document):
        text = document.text.strip()
        if not text:
            return  # Empty is OK for optional fields

        try:
            if self.field_type == "integer":
                value = int(text)
            else:
                value = float(text)

            if self.minimum is not None and value < self.minimum:
                raise ValidationError(
                    message=f"Must be ≥ {self.minimum}", cursor_position=len(text)
                )

            if self.maximum is not None and value > self.maximum:
                raise ValidationError(
                    message=f"Must be ≤ {self.maximum}", cursor_position=len(text)
                )

        except ValueError:
            raise ValidationError(message=f"Invalid {self.field_type}", cursor_position=len(text))


class SimpleStringValidator(Validator):
    """Simple string validator with real-time feedback."""

    def __init__(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
        pattern: str | None = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = re.compile(pattern, re.DOTALL) if pattern else None

    def validate(self, document):
        text = document.text
        if not text:
            return  # Empty is OK for optional fields

        if self.min_length is not None and len(text) < self.min_length:
            raise ValidationError(
                message=f"Need {self.min_length - len(text)} more chars", cursor_position=len(text)
            )

        if self.max_length is not None and len(text) > self.max_length:
            raise ValidationError(
                message=f"Too long by {len(text) - self.max_length} chars",
                cursor_position=self.max_length,
            )

        if self.pattern is not None and self.pattern.fullmatch(text) is None:
            # TODO: Wrap or truncate line if too long
            raise ValidationError(
                message=f"Must match pattern '{self.pattern.pattern}'", cursor_position=len(text)
            )


class FormatValidator(Validator):
    """Format-specific validator using Pydantic validators."""

    _EMAIL_ADAPTER = TypeAdapter(EmailStr)
    _URI_ADAPTER = TypeAdapter(AnyUrl)

    def __init__(self, format_type: str):
        self.format_type = format_type

    def _validate_non_empty_text(self, text: str) -> None:
        if self.format_type == "email":
            self._EMAIL_ADAPTER.validate_python(text)
            return

        if self.format_type == "uri":
            self._URI_ADAPTER.validate_strings(text)
            return

        if self.format_type == "date":
            date.fromisoformat(text)
            return

        if self.format_type == "date-time":
            datetime.fromisoformat(text.replace("Z", "+00:00"))

    def _validation_error_message(self) -> str:
        if self.format_type == "email":
            return "Invalid email format"
        if self.format_type == "uri":
            return "Invalid URI format"
        if self.format_type == "date":
            return "Invalid date (use YYYY-MM-DD)"
        if self.format_type == "date-time":
            return "Invalid datetime (use ISO 8601)"
        return f"Invalid {self.format_type} format"

    def validate(self, document):
        text = document.text.strip()
        if not text:
            return  # Empty is OK for optional fields

        try:
            self._validate_non_empty_text(text)
        except (PydanticValidationError, ValueError):
            raise ValidationError(
                message=self._validation_error_message(),
                cursor_position=len(text),
            )


class ElicitationForm:
    """Simplified elicitation form with all fields visible."""

    def __init__(
        self, schema: ElicitRequestedSchema, message: str, agent_name: str, server_name: str
    ):
        self.schema = schema
        self.message = message
        self.agent_name = agent_name
        self.server_name = server_name

        # Parse schema
        self.properties = schema.get("properties", {})
        self.required_fields = schema.get("required", [])

        # Field storage
        self.field_widgets: dict[str, Any] = {}
        self.multiline_fields: set[str] = set()  # Track which fields are multiline

        # Result
        self.result = None
        self.action = "cancel"

        # Build form
        self._build_form()

    def _build_fastagent_header(self) -> Window:
        fastagent_info = FormattedText(
            [
                ("class:agent-name", self.agent_name),
                ("class:label", " ("),
                ("class:server-name", self.server_name),
                ("class:label", ")"),
            ]
        )
        return Window(FormattedTextControl(fastagent_info), height=1)

    def _build_message_header(self) -> Window:
        mcp_message = FormattedText([("class:message", self.message)])
        return Window(
            FormattedTextControl(mcp_message),
            height=len(self.message.split("\n")),
        )

    def _build_sticky_headers(self) -> HSplit:
        fastagent_header = self._build_fastagent_header()
        mcp_header = self._build_message_header()
        return HSplit(
            [
                VSplit(
                    [
                        Window(width=1),
                        fastagent_header,
                        Window(width=1),
                    ]
                ),
                VSplit(
                    [
                        Window(width=1),
                        mcp_header,
                        Window(width=1),
                    ]
                ),
            ]
        )

    def _build_form_fields(self) -> list[Any]:
        form_fields: list[Any] = []
        for field_index, (field_name, field_def) in enumerate(self.properties.items()):
            field_widget = self._create_field(field_name, field_def)
            if field_widget:
                form_fields.append(field_widget)
                if field_index < len(self.properties) - 1:
                    form_fields.append(Window(height=1))
        return form_fields

    def _build_buttons(self) -> tuple[VSplit, Button]:
        self.status_control = FormattedTextControl(text="")
        self.status_line = Window(self.status_control, height=1)
        submit_btn = Button("Accept", handler=self._accept)
        cancel_btn = Button("Cancel", handler=self._cancel)
        decline_btn = Button("Decline", handler=self._decline)
        cancel_all_btn = Button("Cancel All", handler=self._cancel_all)
        self.buttons = [submit_btn, decline_btn, cancel_btn, cancel_all_btn]
        buttons = VSplit(
            [
                submit_btn,
                Window(width=2),
                decline_btn,
                Window(width=2),
                cancel_btn,
                Window(width=2),
                cancel_all_btn,
            ]
        )
        return buttons, submit_btn

    def _build_scrollable_content(self, buttons: VSplit) -> ScrollablePane:
        form_fields = self._build_form_fields()
        form_fields.extend([self.status_line, buttons])
        scrollable_form_content = HSplit(form_fields)
        padded_scrollable_content = HSplit(
            [
                VSplit(
                    [
                        Window(width=1),
                        scrollable_form_content,
                        Window(width=1),
                    ]
                )
            ]
        )
        return ScrollablePane(
            content=padded_scrollable_content,
            show_scrollbar=False,
            display_arrows=False,
            keep_cursor_visible=True,
            keep_focused_window_visible=True,
        )

    def _dialog_title(self) -> str:
        dialog_title = self.schema.get("title") if isinstance(self.schema, dict) else None
        if not dialog_title or not isinstance(dialog_title, str):
            return "Elicitation Request"
        return dialog_title

    def _build_dialog(self, sticky_headers: HSplit, scrollable_content: ScrollablePane) -> VSplit:
        full_content = HSplit(
            [
                sticky_headers,
                Window(height=1),
                scrollable_content,
            ]
        )
        dialog = Frame(
            body=full_content,
            title=self._dialog_title(),
            style="class:dialog",
        )
        self._dialog = dialog
        return VSplit([Window(width=4), dialog, Window(width=4)])

    def _toolbar_text(self) -> FormattedText:
        if hasattr(self, "_toolbar_hidden") and self._toolbar_hidden:
            return FormattedText([])

        mode_label = "TEXT MODE" if text_navigation_mode else "FIELD MODE"
        mode_color = "ansired" if text_navigation_mode else "ansigreen"
        navigation_tail = (
            " | <CTRL+T> toggle text mode. <TAB> navigate. <ENTER> insert new line."
            if text_navigation_mode
            else (
                " | <CTRL+T> toggle text mode. "
                "<TAB>/↑↓→← navigate. <Ctrl+J> insert new line."
            )
        )
        actions_line = (
            "  <ESC> cancel. <Cancel All> Auto-Cancel further elicitations from this Server."
            if text_navigation_mode
            else (
                "  <ENTER> submit. <ESC> cancel. <Cancel All> Auto-Cancel further "
                "elicitations from this Server."
            )
        )
        return FormattedText(
            [
                ("class:bottom-toolbar.text", actions_line),
                ("", "\n"),
                ("class:bottom-toolbar.text", " | "),
                (f"fg:{mode_color} bg:ansiblack", f" {mode_label} "),
                ("class:bottom-toolbar.text", navigation_tail),
            ]
        )

    def _track_multiline_buffer(self, current_buffer: Buffer) -> None:
        for field_name, widget in self.field_widgets.items():
            if isinstance(widget, Buffer) and widget == current_buffer:
                self.multiline_fields.add(field_name)
                break

    def _add_navigation_key_bindings(self, kb: KeyBindings) -> None:
        @kb.add("tab")
        def focus_next_with_refresh(event):
            focus_next(event)

        @kb.add("s-tab")
        def focus_previous_with_refresh(event):
            focus_previous(event)

        @kb.add("c-t")
        def toggle_text_navigation_mode(event):
            global text_navigation_mode
            text_navigation_mode = not text_navigation_mode
            event.app.invalidate()

        @kb.add("down", filter=Condition(lambda: not text_navigation_mode))
        def focus_next_arrow(event):
            focus_next(event)

        @kb.add("up", filter=Condition(lambda: not text_navigation_mode))
        def focus_previous_arrow(event):
            focus_previous(event)

        @kb.add("right", eager=True, filter=Condition(lambda: not text_navigation_mode))
        def focus_next_right(event):
            focus_next(event)

        @kb.add("left", eager=True, filter=Condition(lambda: not text_navigation_mode))
        def focus_previous_left(event):
            focus_previous(event)

    def _add_submission_key_bindings(self, kb: KeyBindings) -> None:
        @kb.add("c-m", filter=Condition(lambda: not text_navigation_mode))
        def submit_enter(event):
            self._accept()

        @kb.add("c-j", filter=Condition(lambda: not text_navigation_mode))
        def insert_newline_cj(event):
            event.current_buffer.insert_text("\n")
            self._track_multiline_buffer(event.current_buffer)

        @kb.add("c-m", filter=Condition(lambda: text_navigation_mode))
        def insert_newline_enter(event):
            event.current_buffer.insert_text("\n")
            self._track_multiline_buffer(event.current_buffer)

        @kb.add("c-j", filter=Condition(lambda: text_navigation_mode))
        def _(event):
            pass

        @kb.add("escape", eager=True, is_global=True)
        def cancel(event):
            self._cancel()

    def _build_key_bindings(self) -> KeyBindings:
        global text_navigation_mode
        text_navigation_mode = False
        kb = KeyBindings()
        self._add_navigation_key_bindings(kb)
        self._add_submission_key_bindings(kb)
        return kb

    def _build_root_layout(self, constrained_dialog: VSplit) -> HSplit:
        self._get_toolbar = self._toolbar_text
        self._toolbar_window = Window(
            FormattedTextControl(self._toolbar_text),
            height=2,
            style="class:bottom-toolbar",
        )
        root_layout = HSplit([constrained_dialog, self._toolbar_window])
        self._root_layout = root_layout
        return root_layout

    def _set_initial_focus(self, submit_btn: Button) -> None:
        try:
            first_field = None
            for field_name in self.properties.keys():
                widget = self.field_widgets.get(field_name)
                if widget:
                    first_field = widget
                    break

            if first_field:
                self.app.layout.focus(first_field)
            else:
                self.app.layout.focus(submit_btn)
        except Exception:
            pass

    def _build_form(self):
        """Build the form layout."""
        sticky_headers = self._build_sticky_headers()
        buttons, submit_btn = self._build_buttons()
        scrollable_content = self._build_scrollable_content(buttons)
        constrained_dialog = self._build_dialog(sticky_headers, scrollable_content)
        root_layout = self._build_root_layout(constrained_dialog)
        self.app = Application(
            layout=Layout(root_layout),
            key_bindings=self._build_key_bindings(),
            full_screen=False,
            mouse_support=False,
            style=ELICITATION_STYLE,
            include_default_pygments_style=False,
        )
        self.app.invalidate()
        self._set_initial_focus(submit_btn)

    def _extract_enum_schema_options(self, schema_def: dict[str, Any]) -> list[Tuple[str, str]]:
        """Extract options from oneOf/anyOf/enum schema patterns.

        Args:
            schema_def: Schema definition potentially containing oneOf/anyOf/enum

        Returns:
            List of (value, title) tuples for the options
        """
        values = []

        # First check for bare enum (most common pattern for arrays)
        if "enum" in schema_def:
            enum_values = schema_def["enum"]
            enum_names = schema_def.get("enumNames", enum_values)
            for val, name in zip(enum_values, enum_names):
                values.append((val, str(name)))
            return values

        # Then check for oneOf/anyOf patterns
        options = schema_def.get("oneOf", [])
        if not options:
            options = schema_def.get("anyOf", [])

        for option in options:
            if "const" in option:
                value = option["const"]
                title = option.get("title", str(value))
                values.append((value, title))

        return values

    def _extract_string_constraints(self, field_def: dict[str, Any]) -> dict[str, Any]:
        """Extract string constraints from field definition, handling anyOf schemas."""
        constraints = {}

        # Check direct constraints
        if field_def.get("minLength") is not None:
            constraints["minLength"] = field_def["minLength"]
        if field_def.get("maxLength") is not None:
            constraints["maxLength"] = field_def["maxLength"]
        if field_def.get("pattern") is not None:
            constraints["pattern"] = field_def["pattern"]

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
                    break

        return constraints

    def _field_type_hint(self, field_type: str, field_def: dict[str, Any]) -> str | None:
        if field_type != "string":
            return None

        constraints = self._extract_string_constraints(field_def)
        if constraints.get("pattern"):
            return f"Pattern ({constraints['pattern']})"

        format_type = field_def.get("format")
        if not format_type:
            return None

        format_info = {
            "email": ("Email", "user@example.com"),
            "uri": ("URI", "https://example.com"),
            "date": ("Date", "YYYY-MM-DD"),
            "date-time": ("Date Time", "YYYY-MM-DD HH:MM:SS"),
        }
        if format_type in format_info:
            friendly_name, example = format_info[format_type]
            return f"{friendly_name} ({example})"
        return str(format_type)

    def _array_field_hints(self, field_def: dict[str, Any]) -> list[str]:
        hints: list[str] = []
        min_items = field_def.get("minItems")
        max_items = field_def.get("maxItems")
        if min_items is not None and max_items is not None:
            if min_items == max_items:
                hints.append(f"select exactly {min_items}")
            else:
                hints.append(f"select {min_items}-{max_items}")
        elif min_items is not None:
            hints.append(f"select at least {min_items}")
        elif max_items is not None:
            hints.append(f"select up to {max_items}")
        return hints

    def _string_field_hints(self, field_def: dict[str, Any]) -> list[str]:
        hints: list[str] = []
        constraints = self._extract_string_constraints(field_def)
        if constraints.get("minLength"):
            hints.append(f"min {constraints['minLength']} chars")
        if constraints.get("maxLength"):
            hints.append(f"max {constraints['maxLength']} chars")
        return hints

    def _number_field_hints(self, field_def: dict[str, Any]) -> list[str]:
        hints: list[str] = []
        if field_def.get("minimum") is not None:
            hints.append(f"min {field_def['minimum']}")
        if field_def.get("maximum") is not None:
            hints.append(f"max {field_def['maximum']}")
        return hints

    def _field_hints(self, field_type: str, field_def: dict[str, Any]) -> list[str]:
        if field_type == "array" and "items" in field_def:
            return self._array_field_hints(field_def)
        if field_type == "string":
            return self._string_field_hints(field_def)
        if field_type in {"number", "integer"}:
            return self._number_field_hints(field_def)
        return []

    def _build_field_label(self, field_name: str, field_type: str, field_def: dict[str, Any]) -> Label:
        title = field_def.get("title", field_name)
        description = field_def.get("description", "")
        label_text = title + (" *" if field_name in self.required_fields else "")
        if description:
            label_text += f" - {description}"

        hints = self._field_hints(field_type, field_def)
        if hints:
            label_text += f" ({', '.join(hints)})"

        format_hint = self._field_type_hint(field_type, field_def)
        if format_hint:
            return Label(
                text=FormattedText(
                    [
                        ("class:field-label", label_text),
                        ("", "\n"),
                        ("class:field-hint", f"  {format_hint}"),
                    ]
                )
            )
        return Label(text=FormattedText([("class:field-label", label_text)]))

    def _build_boolean_field(self, field_name: str, label: Label, field_def: dict[str, Any]) -> HSplit:
        checkbox = Checkbox(text="Yes")
        checkbox.checked = field_def.get("default", False)
        self.field_widgets[field_name] = checkbox
        return HSplit([label, Frame(checkbox)])

    def _build_enum_radio_field(
        self,
        field_name: str,
        label: Label,
        values: list[Tuple[str, str]],
        *,
        default_value: Any = None,
    ) -> HSplit:
        radio_list = RadioList(values=values, default=default_value)
        self.field_widgets[field_name] = radio_list
        return HSplit([label, Frame(radio_list, height=min(len(values) + 2, 6))])

    def _build_array_field(
        self,
        field_name: str,
        label: Label,
        field_def: dict[str, Any],
    ) -> HSplit | None:
        items_def = field_def["items"]
        values = self._extract_enum_schema_options(items_def)
        if not values:
            return None

        checkbox_list = ValidatedCheckboxList(
            values=values,
            default_values=field_def.get("default", []),
            min_items=field_def.get("minItems"),
            max_items=field_def.get("maxItems"),
        )
        self.field_widgets[field_name] = checkbox_list
        return HSplit([label, Frame(checkbox_list, height=min(len(values) + 2, 8))])

    def _input_validator(self, field_type: str, field_def: dict[str, Any]) -> Validator | None:
        if field_type in {"number", "integer"}:
            return SimpleNumberValidator(
                field_type=field_type,
                minimum=field_def.get("minimum"),
                maximum=field_def.get("maximum"),
            )
        if field_type != "string":
            return None

        constraints = self._extract_string_constraints(field_def)
        format_type = field_def.get("format")
        if format_type in ["email", "uri", "date", "date-time"]:
            return FormatValidator(format_type)
        return SimpleStringValidator(
            min_length=constraints.get("minLength"),
            max_length=constraints.get("maxLength"),
            pattern=constraints.get("pattern"),
        )

    def _multiline_config(self, field_name: str, field_type: str, field_def: dict[str, Any]) -> tuple[bool, int]:
        default_value = field_def.get("default")
        if field_type == "string" and default_value is not None and "\n" in str(default_value):
            self.multiline_fields.add(field_name)
            return True, str(default_value).count("\n") + 1

        max_length = None
        if field_type == "string":
            constraints = self._extract_string_constraints(field_def)
            max_length = constraints.get("maxLength")
            if not max_length and default_value is not None:
                max_length = len(str(default_value))

        if max_length and max_length > 100:
            self.multiline_fields.add(field_name)
            return True, 3 if max_length <= 300 else 5

        return False, 1

    def _build_text_input_field(
        self,
        field_name: str,
        label: Label,
        field_type: str,
        field_def: dict[str, Any],
    ) -> HSplit:
        validator = self._input_validator(field_type, field_def)
        multiline, initial_height = self._multiline_config(field_name, field_type, field_def)
        buffer = Buffer(
            validator=validator,
            multiline=multiline,
            validate_while_typing=True,
            complete_while_typing=False,
            enable_history_search=False,
        )
        default_value = field_def.get("default")
        if default_value is not None:
            buffer.text = str(default_value)
        self.field_widgets[field_name] = buffer
        text_input = Window(
            BufferControl(buffer=buffer),
            height=lambda: self._field_display_height(buffer, initial_height),
            style=lambda: self._field_style(buffer),
            wrap_lines=multiline,
            char=" ",
        )
        input_with_prefix = VSplit(
            [
                Window(width=1, char="▎", style="class:prefix"),
                Window(width=1),
                text_input,
            ]
        )
        return HSplit([label, input_with_prefix])

    def _field_style(self, buffer: Buffer) -> str:
        from prompt_toolkit.application.current import get_app

        if buffer.validation_error:
            return "class:input-field.error"
        if get_app().layout.has_focus(buffer):
            return "class:input-field.focused"
        return "class:input-field"

    def _field_display_height(self, buffer: Buffer, initial_height: int) -> int:
        if not buffer.text:
            return initial_height
        line_count = buffer.text.count("\n") + 1
        return min(max(line_count, initial_height), 20)

    def _create_field(self, field_name: str, field_def: dict[str, Any]):
        """Create a field widget."""
        field_type = field_def.get("type", "string")
        label = self._build_field_label(field_name, field_type, field_def)
        if field_type == "boolean":
            return self._build_boolean_field(field_name, label, field_def)
        if field_type == "string" and "enum" in field_def:
            enum_values = field_def["enum"]
            enum_names = field_def.get("enumNames", enum_values)
            values = [(val, name) for val, name in zip(enum_values, enum_names)]
            return self._build_enum_radio_field(
                field_name,
                label,
                values,
                default_value=field_def.get("default"),
            )
        if field_type == "string" and "oneOf" in field_def:
            values = self._extract_enum_schema_options(field_def)
            if values:
                return self._build_enum_radio_field(
                    field_name,
                    label,
                    values,
                    default_value=field_def.get("default"),
                )
        if field_type == "array" and "items" in field_def:
            return self._build_array_field(field_name, label, field_def)
        return self._build_text_input_field(field_name, label, field_type, field_def)

    def _validation_error_for_widget(
        self,
        field_name: str,
        field_def: dict[str, Any],
        widget: Any,
    ) -> str | None:
        title = field_def.get("title", field_name)
        if isinstance(widget, Buffer) and widget.validation_error:
            return f"'{title}': {widget.validation_error.message}"
        if isinstance(widget, ValidatedCheckboxList) and widget.validation_error:
            return f"'{title}': {widget.validation_error.message}"
        return None

    def _required_widget_missing(self, field_name: str, widget: Any) -> bool:
        if isinstance(widget, Buffer):
            return not widget.text.strip()
        if isinstance(widget, RadioList):
            return widget.current_value is None
        if isinstance(widget, ValidatedCheckboxList):
            return not widget.current_values
        return False

    def _validate_form(self) -> tuple[bool, str | None]:
        """Validate the entire form."""
        for field_name, field_def in self.properties.items():
            widget = self.field_widgets.get(field_name)
            if widget is None:
                continue
            error = self._validation_error_for_widget(field_name, field_def, widget)
            if error is not None:
                return False, error

        for field_name in self.required_fields:
            widget = self.field_widgets.get(field_name)
            if widget is None:
                continue
            if self._required_widget_missing(field_name, widget):
                title = self.properties[field_name].get("title", field_name)
                return False, f"'{title}' is required"

        return True, None

    def _buffer_field_value(self, field_name: str, field_type: str, widget: Buffer) -> Any | None:
        value = widget.text.strip()
        if not value:
            if field_name in self.required_fields:
                return ""
            return None
        if field_type == "integer":
            try:
                return int(value)
            except ValueError as exc:
                raise ValueError(f"Invalid integer value for {field_name}: {value}") from exc
        if field_type == "number":
            try:
                return float(value)
            except ValueError as exc:
                raise ValueError(f"Invalid number value for {field_name}: {value}") from exc
        return value

    def _widget_field_value(self, field_name: str, field_def: dict[str, Any], widget: Any) -> Any:
        field_type = field_def.get("type", "string")
        if isinstance(widget, Buffer):
            return self._buffer_field_value(field_name, field_type, widget)
        if isinstance(widget, Checkbox):
            return widget.checked
        if isinstance(widget, RadioList):
            return widget.current_value
        if isinstance(widget, ValidatedCheckboxList):
            selected_values = widget.current_values
            if selected_values:
                return list(selected_values)
            if field_name not in self.required_fields:
                return []
            return None
        return None

    def _get_form_data(self) -> dict[str, Any]:
        """Extract data from form fields."""
        data: dict[str, Any] = {}
        for field_name, field_def in self.properties.items():
            widget = self.field_widgets.get(field_name)
            if widget is None:
                continue
            value = self._widget_field_value(field_name, field_def, widget)
            if value is None and field_name not in self.required_fields:
                continue
            data[field_name] = value
        return data

    def _accept(self):
        """Handle form submission."""
        # Validate
        is_valid, error_msg = self._validate_form()
        if not is_valid:
            # Use styled error message
            self.status_control.text = FormattedText(
                [("class:validation-error", f"Error: {error_msg}")]
            )
            return

        # Get data
        try:
            self.result = self._get_form_data()
            self.action = "accept"
            self._clear_status_bar()
            self.app.exit()
        except Exception as e:
            # Use styled error message
            self.status_control.text = FormattedText(
                [("class:validation-error", f"Error: {str(e)}")]
            )

    def _cancel(self):
        """Handle cancel."""
        self.action = "cancel"
        self._clear_status_bar()
        self.app.exit()

    def _decline(self):
        """Handle decline."""
        self.action = "decline"
        self._clear_status_bar()
        self.app.exit()

    def _cancel_all(self):
        """Handle cancel all: signal disable; no side effects here.

        UI emits an action; handler/orchestration is responsible for updating state.
        """
        self.action = "disable"
        self._clear_status_bar()
        self.app.exit()

    def _clear_status_bar(self):
        """Hide the status bar by removing it from the layout."""
        # Create completely clean layout - just empty space with application background
        from prompt_toolkit.layout import HSplit, Window
        from prompt_toolkit.layout.controls import FormattedTextControl

        # Create a simple empty window with application background
        empty_window = Window(
            FormattedTextControl(FormattedText([("class:application", "")])), height=1
        )

        # Replace entire layout with just the empty window
        new_layout = HSplit([empty_window])

        # Update the app's layout
        if hasattr(self, "app") and self.app:
            self.app.layout.container = new_layout
            self.app.invalidate()

    async def run_async(self) -> tuple[str, dict[str, Any] | None]:
        """Run the form and return result."""
        try:
            with suppress_known_runtime_warnings():
                await self.app.run_async()
        except Exception as e:
            print(f"Form error: {e}")
            self.action = "cancel"
            self._clear_status_bar()
        return self.action, self.result


async def show_simple_elicitation_form(
    schema: ElicitRequestedSchema, message: str, agent_name: str, server_name: str
) -> tuple[str, dict[str, Any] | None]:
    """Show the simplified elicitation form."""
    form = ElicitationForm(schema, message, agent_name, server_name)
    return await form.run_async()
