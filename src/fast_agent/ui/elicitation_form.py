"""Simplified, robust elicitation form dialog."""

import re
from datetime import date, datetime
from typing import Any, Dict, Optional

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
from pydantic import AnyUrl, EmailStr
from pydantic import ValidationError as PydanticValidationError

from fast_agent.ui.elicitation_style import ELICITATION_STYLE

text_navigation_mode = False


class SimpleNumberValidator(Validator):
    """Simple number validator with real-time feedback."""

    def __init__(
        self, field_type: str, minimum: Optional[float] = None, maximum: Optional[float] = None
    ):
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
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
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

    def __init__(self, format_type: str):
        self.format_type = format_type

    def validate(self, document):
        text = document.text.strip()
        if not text:
            return  # Empty is OK for optional fields

        try:
            if self.format_type == "email":
                # Use Pydantic model validation for email
                from pydantic import BaseModel

                class EmailModel(BaseModel):
                    email: EmailStr

                EmailModel(email=text)
            elif self.format_type == "uri":
                # Use Pydantic model validation for URI
                from pydantic import BaseModel

                class UriModel(BaseModel):
                    uri: AnyUrl

                UriModel(uri=text)
            elif self.format_type == "date":
                # Validate ISO date format (YYYY-MM-DD)
                date.fromisoformat(text)
            elif self.format_type == "date-time":
                # Validate ISO datetime format
                datetime.fromisoformat(text.replace("Z", "+00:00"))
        except (PydanticValidationError, ValueError):
            # Extract readable error message
            if self.format_type == "email":
                message = "Invalid email format"
            elif self.format_type == "uri":
                message = "Invalid URI format"
            elif self.format_type == "date":
                message = "Invalid date (use YYYY-MM-DD)"
            elif self.format_type == "date-time":
                message = "Invalid datetime (use ISO 8601)"
            else:
                message = f"Invalid {self.format_type} format"

            raise ValidationError(message=message, cursor_position=len(text))


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

    def _build_form(self):
        """Build the form layout."""

        # Fast-agent provided data (Agent and MCP Server) - aligned labels
        fastagent_info = FormattedText(
            [
                ("class:label", "Agent:      "),
                ("class:agent-name", self.agent_name),
                ("class:label", "\nMCP Server: "),
                ("class:server-name", self.server_name),
            ]
        )
        fastagent_header = Window(
            FormattedTextControl(fastagent_info),
            height=2,  # Just agent and server lines
        )

        # MCP Server provided message
        mcp_message = FormattedText([("class:message", self.message)])
        mcp_header = Window(
            FormattedTextControl(mcp_message),
            height=len(self.message.split("\n")),
        )

        # Create sticky headers (outside scrollable area)
        sticky_headers = HSplit(
            [
                Window(height=1),  # Top padding
                VSplit(
                    [
                        Window(width=2),  # Left padding
                        fastagent_header,  # Fast-agent info
                        Window(width=2),  # Right padding
                    ]
                ),
                Window(height=1),  # Spacing
                VSplit(
                    [
                        Window(width=2),  # Left padding
                        mcp_header,  # MCP server message
                        Window(width=2),  # Right padding
                    ]
                ),
                Window(height=1),  # Spacing
            ]
        )

        # Create scrollable form fields (without headers)
        form_fields = []

        for field_name, field_def in self.properties.items():
            field_widget = self._create_field(field_name, field_def)
            if field_widget:
                form_fields.append(field_widget)
                form_fields.append(Window(height=1))  # Spacing

        # Status line for error display (disabled ValidationToolbar to avoid confusion)
        self.status_control = FormattedTextControl(text="")
        self.status_line = Window(
            self.status_control, height=1
        )  # Store reference for later clearing

        # Buttons - ensure they accept focus
        submit_btn = Button("Accept", handler=self._accept)
        cancel_btn = Button("Cancel", handler=self._cancel)
        decline_btn = Button("Decline", handler=self._decline)
        cancel_all_btn = Button("Cancel All", handler=self._cancel_all)

        # Store button references for focus debugging
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

        # Main scrollable content (form fields and buttons only)
        form_fields.extend([self.status_line, buttons])
        scrollable_form_content = HSplit(form_fields)

        # Add padding around scrollable content
        padded_scrollable_content = HSplit(
            [
                VSplit(
                    [
                        Window(width=2),  # Left padding
                        scrollable_form_content,
                        Window(width=2),  # Right padding
                    ]
                ),
                Window(height=1),  # Bottom padding
            ]
        )

        # Wrap only form fields in ScrollablePane (headers stay fixed)
        scrollable_content = ScrollablePane(
            content=padded_scrollable_content,
            show_scrollbar=False,  # Only show when content exceeds available space
            display_arrows=False,  # Only show when content exceeds available space
            keep_cursor_visible=True,
            keep_focused_window_visible=True,
        )

        # Combine sticky headers and scrollable content (no separate title bar needed)
        full_content = HSplit(
            [
                Window(height=1),  # Top spacing
                sticky_headers,  # Headers stay fixed at top
                scrollable_content,  # Form fields can scroll
            ]
        )

        # Choose dialog title: prefer schema.title if provided
        dialog_title = self.schema.get("title") if isinstance(self.schema, dict) else None
        if not dialog_title or not isinstance(dialog_title, str):
            dialog_title = "Elicitation Request"

        # Create dialog frame with dynamic title
        dialog = Frame(
            body=full_content,
            title=dialog_title,
            style="class:dialog",
        )

        # Apply width constraints by putting Frame in VSplit with flexible spacers
        # This prevents console display interference and constrains the Frame border
        constrained_dialog = VSplit(
            [
                Window(width=10),  # Smaller left spacer
                dialog,
                Window(width=10),  # Smaller right spacer
            ]
        )

        # Use field navigation mode as default
        global text_navigation_mode
        text_navigation_mode = False

        # Key bindings
        kb = KeyBindings()

        @kb.add("tab")
        def focus_next_with_refresh(event):
            focus_next(event)

        @kb.add("s-tab")
        def focus_previous_with_refresh(event):
            focus_previous(event)

        # Toggle between text navigation mode and field navigation mode
        @kb.add("c-t")
        def toggle_text_navigation_mode(event):
            global text_navigation_mode
            text_navigation_mode = not text_navigation_mode
            event.app.invalidate()  # Force redraw the app to update toolbar

        # Arrow key navigation - let radio lists handle up/down first
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

        # Enter submits in field navigation mode
        @kb.add("c-m", filter=Condition(lambda: not text_navigation_mode))
        def submit_enter(event):
            self._accept()

        # Ctrl+J inserts newlines in field navigation mode
        @kb.add("c-j", filter=Condition(lambda: not text_navigation_mode))
        def insert_newline_cj(event):
            # Insert a newline at the cursor position
            event.current_buffer.insert_text("\n")
            # Mark this field as multiline when user adds a newline
            for field_name, widget in self.field_widgets.items():
                if isinstance(widget, Buffer) and widget == event.current_buffer:
                    self.multiline_fields.add(field_name)
                    break

        # Enter inserts new lines in text navigation mode
        @kb.add("c-m", filter=Condition(lambda: text_navigation_mode))
        def insert_newline_enter(event):
            # Insert a newline at the cursor position
            event.current_buffer.insert_text("\n")
            # Mark this field as multiline when user adds a newline
            for field_name, widget in self.field_widgets.items():
                if isinstance(widget, Buffer) and widget == event.current_buffer:
                    self.multiline_fields.add(field_name)
                    break

        # deactivate ctrl+j in text navigation mode
        @kb.add("c-j", filter=Condition(lambda: text_navigation_mode))
        def _(event):
            pass

        # ESC should ALWAYS cancel immediately, no matter what
        @kb.add("escape", eager=True, is_global=True)
        def cancel(event):
            self._cancel()

        # Create a root layout with the dialog and bottom toolbar
        def get_toolbar():
            # When clearing, return empty to hide the toolbar completely
            if hasattr(self, "_toolbar_hidden") and self._toolbar_hidden:
                return FormattedText([])

            mode_label = "TEXT MODE" if text_navigation_mode else "FIELD MODE"
            mode_color = "ansired" if text_navigation_mode else "ansigreen"

            arrow_up = "↑"
            arrow_down = "↓"
            arrow_left = "←"
            arrow_right = "→"

            if text_navigation_mode:
                actions_line = (
                    "  <ESC> cancel. <Cancel All> Auto-Cancel further elicitations from this Server."
                )
                navigation_tail = (
                    " | <CTRL+T> toggle text mode. <TAB> navigate. <ENTER> insert new line."
                )
            else:
                actions_line = (
                    "  <ENTER> submit. <ESC> cancel. <Cancel All> Auto-Cancel further elicitations "
                    "from this Server."
                )
                navigation_tail = (
                    " | <CTRL+T> toggle text mode. "
                    f"<TAB>/{arrow_up}{arrow_down}{arrow_right}{arrow_left} navigate. "
                    "<Ctrl+J> insert new line."
                )

            formatted_segments = [
                ("class:bottom-toolbar.text", actions_line),
                ("", "\n"),
                ("class:bottom-toolbar.text", " | "),
                (f"fg:{mode_color} bg:ansiblack", f" {mode_label} "),
                ("class:bottom-toolbar.text", navigation_tail),
            ]
            return FormattedText(formatted_segments)

        # Store toolbar function reference for later control
        self._get_toolbar = get_toolbar
        self._dialog = dialog

        # Create toolbar window that we can reference later
        self._toolbar_window = Window(
            FormattedTextControl(get_toolbar), height=2, style="class:bottom-toolbar"
        )

        # Add toolbar to the layout
        root_layout = HSplit(
            [
                constrained_dialog,  # The width-constrained dialog
                self._toolbar_window,
            ]
        )
        self._root_layout = root_layout

        # Application with toolbar and validation - ensure our styles override defaults
        self.app = Application(
            layout=Layout(root_layout),
            key_bindings=kb,
            full_screen=False,  # Back to windowed mode for better integration
            mouse_support=False,
            style=ELICITATION_STYLE,
            include_default_pygments_style=False,  # Use only our custom style
        )

        # Set initial focus to first form field
        def set_initial_focus():
            try:
                # Find first form field to focus on
                first_field = None
                for field_name in self.properties.keys():
                    widget = self.field_widgets.get(field_name)
                    if widget:
                        first_field = widget
                        break

                if first_field:
                    self.app.layout.focus(first_field)
                else:
                    # Fallback to first button if no fields
                    self.app.layout.focus(submit_btn)
            except Exception:
                pass  # If focus fails, continue without it

        # Schedule focus setting for after layout is ready
        self.app.invalidate()  # Ensure layout is built
        set_initial_focus()

    def _extract_string_constraints(self, field_def: Dict[str, Any]) -> Dict[str, Any]:
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

    def _create_field(self, field_name: str, field_def: Dict[str, Any]):
        """Create a field widget."""

        field_type = field_def.get("type", "string")
        title = field_def.get("title", field_name)
        description = field_def.get("description", "")
        is_required = field_name in self.required_fields

        # Build label with validation hints
        label_text = title
        if is_required:
            label_text += " *"
        if description:
            label_text += f" - {description}"

        # Add validation hints (simple ones stay on same line)
        hints = []
        format_hint = None

        if field_type == "string":
            constraints = self._extract_string_constraints(field_def)
            if constraints.get("minLength"):
                hints.append(f"min {constraints['minLength']} chars")
            if constraints.get("maxLength"):
                hints.append(f"max {constraints['maxLength']} chars")

            if constraints.get("pattern"):
                # TODO: Wrap or truncate line if too long
                format_hint = f"Pattern: {constraints['pattern']}"

            # Handle format hints separately (these go on next line)
            format_type = field_def.get("format")
            if format_type:
                format_info = {
                    "email": ("Email", "user@example.com"),
                    "uri": ("URI", "https://example.com"),
                    "date": ("Date", "YYYY-MM-DD"),
                    "date-time": ("Date Time", "YYYY-MM-DD HH:MM:SS"),
                }
                if format_type in format_info:
                    friendly_name, example = format_info[format_type]
                    format_hint = f"{friendly_name}: {example}"
                else:
                    format_hint = format_type

        elif field_type in ["number", "integer"]:
            if field_def.get("minimum") is not None:
                hints.append(f"min {field_def['minimum']}")
            if field_def.get("maximum") is not None:
                hints.append(f"max {field_def['maximum']}")
        elif field_type == "string" and "enum" in field_def:
            enum_names = field_def.get("enumNames", field_def["enum"])
            hints.append(f"choose from: {', '.join(enum_names)}")

        # Add simple hints to main label line
        if hints:
            label_text += f" ({', '.join(hints)})"

        # Create multiline label if we have format hints
        if format_hint:
            label_lines = [label_text, f"  → {format_hint}"]
            label = Label(text="\n".join(label_lines))
        else:
            label = Label(text=label_text)

        # Create input widget based on type
        if field_type == "boolean":
            default = field_def.get("default", False)
            checkbox = Checkbox(text="Yes")
            checkbox.checked = default
            self.field_widgets[field_name] = checkbox

            return HSplit([label, Frame(checkbox)])

        elif field_type == "string" and "enum" in field_def:
            enum_values = field_def["enum"]
            enum_names = field_def.get("enumNames", enum_values)
            values = [(val, name) for val, name in zip(enum_values, enum_names)]

            default_value = field_def.get("default")
            radio_list = RadioList(values=values, default=default_value)
            self.field_widgets[field_name] = radio_list

            return HSplit([label, Frame(radio_list, height=min(len(values) + 2, 6))])

        else:
            # Text/number input
            validator: Validator | None = None

            if field_type in ["number", "integer"]:
                validator = SimpleNumberValidator(
                    field_type=field_type,
                    minimum=field_def.get("minimum"),
                    maximum=field_def.get("maximum"),
                )
            elif field_type == "string":
                constraints = self._extract_string_constraints(field_def)
                format_type = field_def.get("format")

                if format_type in ["email", "uri", "date", "date-time"]:
                    # Use format validator for specific formats
                    validator = FormatValidator(format_type)
                else:
                    # Use string length validator for regular strings
                    validator = SimpleStringValidator(
                        min_length=constraints.get("minLength"),
                        max_length=constraints.get("maxLength"),
                        pattern=constraints.get("pattern"),
                    )
            else:
                constraints = {}

            default_value = field_def.get("default")

            # Determine if field should be multiline based on max_length or default value length
            if field_type == "string":
                max_length = constraints.get("maxLength")
                # Check default value length if maxLength not specified
                if not max_length and default_value is not None:
                    max_length = len(str(default_value))
            else:
                max_length = None

            # Check if default value contains newlines
            if field_type == "string" and default_value is not None and "\n" in str(default_value):
                multiline = True
                self.multiline_fields.add(field_name)  # Track multiline fields
                # Set height to actual line count for fields with newlines in default
                initial_height = str(default_value).count("\n") + 1
            elif max_length and max_length > 100:
                # Use multiline for longer fields
                multiline = True
                self.multiline_fields.add(field_name)  # Track multiline fields
                if max_length <= 300:
                    initial_height = 3
                else:
                    initial_height = 5
            else:
                # Single line for shorter fields
                multiline = False
                initial_height = 1

            buffer = Buffer(
                validator=validator,
                multiline=multiline,
                validate_while_typing=True,  # Enable real-time validation
                complete_while_typing=False,  # Disable completion for cleaner experience
                enable_history_search=False,  # Disable history for cleaner experience
            )
            if default_value is not None:
                buffer.text = str(default_value)
            self.field_widgets[field_name] = buffer

            # Create dynamic style function for focus highlighting and validation errors
            def get_field_style():
                """Dynamic style that changes based on focus and validation state."""
                from prompt_toolkit.application.current import get_app

                # Check if buffer has validation errors
                if buffer.validation_error:
                    return "class:input-field.error"
                elif get_app().layout.has_focus(buffer):
                    return "class:input-field.focused"
                else:
                    return "class:input-field"

            # Create a dynamic height function based on content
            def get_dynamic_height():
                if not buffer.text:
                    return initial_height
                # Calculate height based on number of newlines in buffer
                line_count = buffer.text.count("\n") + 1
                # Use initial height as minimum, grow up to 20 lines
                return min(max(line_count, initial_height), 20)

            text_input = Window(
                BufferControl(buffer=buffer),
                height=get_dynamic_height,  # Use dynamic height function
                style=get_field_style,  # Use dynamic style function
                wrap_lines=True if multiline else False,  # Enable word wrap for multiline
            )

            return HSplit([label, Frame(text_input)])

    def _validate_form(self) -> tuple[bool, Optional[str]]:
        """Validate the entire form."""

        # First, check all fields for validation errors from their validators
        for field_name, field_def in self.properties.items():
            widget = self.field_widgets.get(field_name)
            if widget is None:
                continue

            # Check for validation errors from validators
            if isinstance(widget, Buffer):
                if widget.validation_error:
                    title = field_def.get("title", field_name)
                    return False, f"'{title}': {widget.validation_error.message}"

        # Then check if required fields are empty
        for field_name in self.required_fields:
            widget = self.field_widgets.get(field_name)
            if widget is None:
                continue

            # Check if required field has value
            if isinstance(widget, Buffer):
                if not widget.text.strip():
                    title = self.properties[field_name].get("title", field_name)
                    return False, f"'{title}' is required"
            elif isinstance(widget, RadioList):
                if widget.current_value is None:
                    title = self.properties[field_name].get("title", field_name)
                    return False, f"'{title}' is required"

        return True, None

    def _get_form_data(self) -> Dict[str, Any]:
        """Extract data from form fields."""
        data: Dict[str, Any] = {}

        for field_name, field_def in self.properties.items():
            widget = self.field_widgets.get(field_name)
            if widget is None:
                continue

            field_type = field_def.get("type", "string")

            if isinstance(widget, Buffer):
                value = widget.text.strip()
                if value:
                    if field_type == "integer":
                        try:
                            data[field_name] = int(value)
                        except ValueError:
                            # This should not happen due to validation, but be safe
                            raise ValueError(f"Invalid integer value for {field_name}: {value}")
                    elif field_type == "number":
                        try:
                            data[field_name] = float(value)
                        except ValueError:
                            # This should not happen due to validation, but be safe
                            raise ValueError(f"Invalid number value for {field_name}: {value}")
                    else:
                        data[field_name] = value
                elif field_name not in self.required_fields:
                    data[field_name] = None

            elif isinstance(widget, Checkbox):
                data[field_name] = widget.checked

            elif isinstance(widget, RadioList):
                if widget.current_value is not None:
                    data[field_name] = widget.current_value

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

    async def run_async(self) -> tuple[str, Optional[Dict[str, Any]]]:
        """Run the form and return result."""
        try:
            await self.app.run_async()
        except Exception as e:
            print(f"Form error: {e}")
            self.action = "cancel"
            self._clear_status_bar()
        return self.action, self.result


async def show_simple_elicitation_form(
    schema: ElicitRequestedSchema, message: str, agent_name: str, server_name: str
) -> tuple[str, Optional[Dict[str, Any]]]:
    """Show the simplified elicitation form."""
    form = ElicitationForm(schema, message, agent_name, server_name)
    return await form.run_async()
