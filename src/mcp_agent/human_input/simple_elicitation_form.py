"""Simplified, robust elicitation form dialog."""

from typing import Any, Dict, Optional
from mcp.types import ElicitRequestedSchema
from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl, BufferControl
from prompt_toolkit.widgets import (
    Label,
    Button,
    RadioList,
    Checkbox,
    Frame,
    Dialog,
    ValidationToolbar,
)
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.key_binding.bindings.focus import focus_next, focus_previous
from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.validation import Validator, ValidationError

from mcp_agent.human_input.elicitation_forms import ELICITATION_STYLE
from mcp_agent.human_input.elicitation_state import elicitation_state


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

    def __init__(self, min_length: Optional[int] = None, max_length: Optional[int] = None):
        self.min_length = min_length
        self.max_length = max_length

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


class SimpleElicitationForm:
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
        self.field_widgets = {}

        # Result
        self.result = None
        self.action = "cancel"

        # Build form
        self._build_form()

    def _build_form(self):
        """Build the form layout."""

        # Fast-agent provided data (Agent and Server)
        fastagent_info = FormattedText(
            [
                ("class:label", "Agent: "),
                ("class:agent-name", self.agent_name),
                ("class:label", "\nServer: "),
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

        # Separator between MCP server content and form fields
        separator = Window(
            FormattedTextControl(FormattedText([("class:separator", "─" * 50)])), height=1
        )

        # Create form fields - correct order now
        form_fields = [
            fastagent_header,  # Fast-agent info
            Window(height=1),  # Spacing
            mcp_header,  # MCP server message
            Window(height=1),  # Spacing
            separator,  # Separator AFTER MCP content
            Window(height=1),  # Spacing
        ]

        for field_name, field_def in self.properties.items():
            field_widget = self._create_field(field_name, field_def)
            if field_widget:
                form_fields.append(field_widget)
                form_fields.append(Window(height=1))  # Spacing

        # Status line for error display (disabled ValidationToolbar to avoid confusion)
        self.status_control = FormattedTextControl(text="")
        status_line = Window(self.status_control, height=1)

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

        # Main layout 
        form_fields.extend([status_line, buttons])
        content = HSplit(form_fields)

        # Add padding around content using HSplit and VSplit with empty windows
        padded_content = HSplit(
            [
                Window(height=1),  # Top padding
                VSplit(
                    [
                        Window(width=2),  # Left padding
                        content,
                        Window(width=2),  # Right padding
                    ]
                ),
                Window(height=1),  # Bottom padding
            ]
        )

        # Dialog
        dialog = Dialog(
            title="Elicitation Form",
            body=padded_content,
            with_background=True,
        )

        # Key bindings
        kb = KeyBindings()
        
        @kb.add("tab")
        def focus_next_with_refresh(event):
            focus_next(event)
            event.app.invalidate()  # Force refresh for focus highlighting
            
        @kb.add("s-tab")
        def focus_previous_with_refresh(event):
            focus_previous(event)
            event.app.invalidate()  # Force refresh for focus highlighting

        @kb.add("c-m")  # Ctrl+Enter
        def submit(event):
            self._accept()

        @kb.add("escape")
        def cancel(event):
            self._cancel()

        # Create a root layout with the dialog and bottom toolbar
        def get_toolbar():
            return FormattedText(
                [
                    ("class:bottom-toolbar.text", " <TAB> to change fields. "),
                    (
                        "class:bottom-toolbar.text",
                        "<Cancel All> Cancel all further elicitations from this MCP Server.",
                    ),
                ]
            )

        # Add toolbar to the layout
        root_layout = HSplit(
            [
                dialog,  # The main dialog
                Window(FormattedTextControl(get_toolbar), height=1, style="class:bottom-toolbar"),
            ]
        )

        # Application with toolbar and validation - ensure our styles override defaults
        self.app = Application(
            layout=Layout(root_layout),
            key_bindings=kb,
            full_screen=False,
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
            except:
                pass  # If focus fails, continue without it

        # Schedule focus setting for after layout is ready
        self.app.invalidate()  # Ensure layout is built
        set_initial_focus()

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

        # Add validation hints
        hints = []
        if field_type == "string":
            if field_def.get("minLength"):
                hints.append(f"min {field_def['minLength']} chars")
            if field_def.get("maxLength"):
                hints.append(f"max {field_def['maxLength']} chars")
        elif field_type in ["number", "integer"]:
            if field_def.get("minimum") is not None:
                hints.append(f"min {field_def['minimum']}")
            if field_def.get("maximum") is not None:
                hints.append(f"max {field_def['maximum']}")
        elif field_type == "string" and "enum" in field_def:
            enum_names = field_def.get("enumNames", field_def["enum"])
            hints.append(f"choose from: {', '.join(enum_names)}")

        if hints:
            label_text += f" ({', '.join(hints)})"

        label = Label(text=label_text)

        # Create input widget based on type
        if field_type == "boolean":
            default = field_def.get("default", False)
            checkbox = Checkbox(text="Yes")
            checkbox.checked = default
            self.field_widgets[field_name] = checkbox

            return HSplit([label, checkbox])

        elif field_type == "string" and "enum" in field_def:
            enum_values = field_def["enum"]
            enum_names = field_def.get("enumNames", enum_values)
            values = [(val, name) for val, name in zip(enum_values, enum_names)]

            radio_list = RadioList(values=values)
            self.field_widgets[field_name] = radio_list

            return HSplit([label, Frame(radio_list, height=min(len(values) + 2, 6))])

        else:
            # Text/number input
            validator = None

            if field_type in ["number", "integer"]:
                validator = SimpleNumberValidator(
                    field_type=field_type,
                    minimum=field_def.get("minimum"),
                    maximum=field_def.get("maximum"),
                )
            elif field_type == "string":
                validator = SimpleStringValidator(
                    min_length=field_def.get("minLength"),
                    max_length=field_def.get("maxLength"),
                )

            buffer = Buffer(
                validator=validator,
                multiline=False,
                validate_while_typing=True,  # Enable real-time validation
                complete_while_typing=False,  # Disable completion for cleaner experience
                enable_history_search=False,  # Disable history for cleaner experience
            )
            self.field_widgets[field_name] = buffer

            # Create dynamic style function for focus highlighting
            def get_field_style():
                """Dynamic style that changes based on focus."""
                from prompt_toolkit.application.current import get_app
                if get_app().layout.has_focus(buffer):
                    return "class:input-field.focused"
                else:
                    return "class:input-field"
            
            text_input = Window(
                BufferControl(buffer=buffer),
                height=1,
                style=get_field_style,  # Use dynamic style function
            )

            return HSplit([label, Frame(text_input)])

    def _validate_form(self) -> tuple[bool, Optional[str]]:
        """Validate the entire form."""

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
        data = {}

        for field_name, field_def in self.properties.items():
            widget = self.field_widgets.get(field_name)
            if widget is None:
                continue

            field_type = field_def.get("type", "string")

            if isinstance(widget, Buffer):
                value = widget.text.strip()
                if value:
                    if field_type == "integer":
                        data[field_name] = int(value)
                    elif field_type == "number":
                        data[field_name] = float(value)
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
            self.app.exit()
        except Exception as e:
            # Use styled error message
            self.status_control.text = FormattedText(
                [("class:validation-error", f"Error: {str(e)}")]
            )

    def _cancel(self):
        """Handle cancel."""
        self.action = "cancel"
        self.app.exit()

    def _decline(self):
        """Handle decline."""
        self.action = "decline"
        self.app.exit()

    def _cancel_all(self):
        """Handle cancel all - cancels and disables future elicitations."""
        elicitation_state.disable_server(self.server_name)
        self.action = "disable"
        self.app.exit()

    async def run_async(self) -> tuple[str, Optional[Dict[str, Any]]]:
        """Run the form and return result."""
        try:
            await self.app.run_async()
        except Exception as e:
            print(f"Form error: {e}")
            self.action = "cancel"
        return self.action, self.result


async def show_simple_elicitation_form(
    schema: ElicitRequestedSchema, message: str, agent_name: str, server_name: str
) -> tuple[str, Optional[Dict[str, Any]]]:
    """Show the simplified elicitation form."""
    form = SimpleElicitationForm(schema, message, agent_name, server_name)
    return await form.run_async()
