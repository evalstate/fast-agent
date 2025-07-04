"""Schema-driven form UI for MCP elicitations using prompt_toolkit."""

from typing import Any, Dict, List, Optional

from mcp.types import ElicitRequestedSchema
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import button_dialog, input_dialog, radiolist_dialog, yes_no_dialog
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import ValidationError, Validator

# Define consistent elicitation style - inspired by usage display and interactive prompt
ELICITATION_STYLE = Style.from_dict(
    {
        # Dialog structure - very dark gray background
        "dialog": "bg:#1a1a1a",  # Very dark gray background
        "dialog.body": "bg:#1a1a1a fg:ansiwhite",
        "dialog.title": "bg:ansiblue fg:ansiwhite bold",  # Blue title bar
        "dialog shadow": "bg:ansiblack",
        
        # Buttons - only define focused state to preserve focus highlighting
        "button.focused": "bg:ansibrightgreen fg:ansiblack bold",  # Bright green with black text for contrast
        "button.arrow": "fg:ansiwhite bold",  # White arrows for visibility
        
        # Form elements with beautiful dark theme
        # Checkboxes - keeping green for boolean on/off clarity
        "checkbox": "fg:#888888",  # Gray unchecked checkbox
        "checkbox-checked": "fg:ansibrightgreen bold",  # Green when checked (on/off clarity)
        "checkbox-selected": "bg:#3a3a3a fg:ansibrightgreen bold",  # Green highlighted when focused
        
        # Radio list styling - matching the dark/gold theme
        "radio-list": "bg:#2a2a2a",  # Dark gray background for radio list
        "radio": "fg:#888888",  # Gray for unselected items
        "radio-selected": "bg:#3a3a3a fg:#ffff66 bold",  # Highlighted item (focused)
        "radio-checked": "fg:#ffcc00 bold",  # Gold for selected item
        
        # Text input areas - now with working style parameter!
        "input-field": "bg:#2a2a2a fg:#ffcc00 bold",  # Dark gray bg, bright gold text
        "input-field.focused": "bg:#3a3a3a fg:#ffff66 bold",  # Lighter gray bg, brighter gold when focused
        
        # Frame styling with ANSI colors
        "frame.border": "fg:#444444",  # Subtle gray borders
        "frame.label": "fg:ansigray",  # Gray frame labels (less prominent)
        
        # Labels and text - less prominent as requested
        "label": "fg:ansigray",  # Gray labels (less prominent)
        "message": "fg:ansibrightcyan",  # Bright cyan messages (no bold)
        
        # Agent and server names - consistent colors
        "agent-name": "fg:ansibrightblue bold",
        "server-name": "fg:ansibrightcyan bold", 
        
        # Validation errors - better contrast  
        "validation-toolbar": "bg:ansibrightred fg:ansiwhite bold",
        "validation-toolbar.text": "bg:ansibrightred fg:ansiwhite",
        "validation.border": "fg:ansibrightred",
        "validation-error": "fg:ansibrightred bold",  # For status line errors
        
        # Separator styling
        "separator": "fg:ansibrightblue bold",
        
        # Completion menu - exactly matching enhanced_prompt.py
        "completion-menu.completion": "bg:ansiblack fg:ansigreen",
        "completion-menu.completion.current": "bg:ansiblack fg:ansigreen bold", 
        "completion-menu.meta.completion": "bg:ansiblack fg:ansiblue",
        "completion-menu.meta.completion.current": "bg:ansibrightblack fg:ansiblue",
        
        # Toolbar - matching enhanced_prompt.py exactly
        "bottom-toolbar": "fg:ansiblack bg:ansigray",
        "bottom-toolbar.text": "fg:ansiblack bg:ansigray",
    }
)


class NumberValidator(Validator):
    """Validator for number fields."""

    def __init__(
        self, field_type: str, minimum: Optional[float] = None, maximum: Optional[float] = None
    ):
        self.field_type = field_type
        self.minimum = minimum
        self.maximum = maximum

    def validate(self, document):
        text = document.text

        if not text and text != "0":  # Allow "0" but not empty
            return  # Empty is OK, will be handled by required check

        try:
            if self.field_type == "integer":
                value = int(text)
            else:
                value = float(text)

            if self.minimum is not None and value < self.minimum:
                raise ValidationError(message=f"Minimum value is {self.minimum}")

            if self.maximum is not None and value > self.maximum:
                raise ValidationError(message=f"Maximum value is {self.maximum}")

        except ValueError:
            if self.field_type == "integer":
                raise ValidationError(message="Please enter a valid integer")
            else:
                raise ValidationError(message="Please enter a valid number")


class StringValidator(Validator):
    """Validator for string fields."""

    def __init__(
        self,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        enum_values: Optional[List[str]] = None,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.enum_values = enum_values

    def validate(self, document):
        text = document.text

        if not text:
            return  # Empty is OK, will be handled by required check

        if self.min_length is not None and len(text) < self.min_length:
            raise ValidationError(message=f"Minimum length is {self.min_length}")

        if self.max_length is not None and len(text) > self.max_length:
            raise ValidationError(message=f"Maximum length is {self.max_length}")

        if self.enum_values is not None and text not in self.enum_values:
            raise ValidationError(message=f"Value must be one of: {', '.join(self.enum_values)}")


async def show_action_menu(agent_name: str, server_name: str, has_schema: bool = False) -> str:
    """Show the initial action selection menu."""

    values = [
        ("accept", HTML("<ansigreen>✓</ansigreen> Accept - Show form with all fields")),
        ("decline", HTML("<ansired>✗</ansired> Decline - Refuse this request")),
        ("cancel", HTML("<ansiyellow>○</ansiyellow> Cancel - Skip this request")),
        (
            "disable",
            HTML(
                "<ansimagenta>⊘</ansimagenta> Disable - Never ask again for {}".format(server_name)
            ),
        ),
    ]

    # Add option for sequential form if schema is available
    if has_schema:
        values.insert(
            1,
            (
                "accept_sequential",
                HTML("<ansigreen>✓</ansigreen> Accept Sequential - Fill fields one at a time"),
            ),
        )

    result = await radiolist_dialog(
        title=HTML('<style bg="#ff4757" fg="#ffffff">Elicitation Request</style>'),
        text=HTML(
            f"<b>Agent:</b> {agent_name}\n<b>Server:</b> {server_name}\n\nChoose your response:"
        ),
        values=values,
        style=ELICITATION_STYLE,
    ).run_async()

    return result or "cancel"


async def build_field_input(
    field_name: str, field_def: Dict[str, Any], required_fields: List[str]
) -> Optional[Any]:
    """Build and show input dialog for a single field."""

    field_type = field_def.get("type", "string")
    title = field_def.get("title", field_name)
    description = field_def.get("description", "")
    is_required = field_name in required_fields

    # Build the prompt text
    prompt_parts = []
    if description:
        prompt_parts.append(description)
    if is_required:
        prompt_parts.append("[Required]")

    prompt_text = "\n".join(prompt_parts) if prompt_parts else None

    # Handle different field types
    if field_type == "boolean":
        # For boolean fields, use yes/no dialog
        default = field_def.get("default", False)
        result = await yes_no_dialog(
            title=title,
            text=prompt_text or "Select Yes or No",
            yes_text="Yes (True)",
            no_text="No (False)",
            style=ELICITATION_STYLE,
        ).run_async()

        if result is None:  # User cancelled
            return None
        return result

    elif field_type == "string" and "enum" in field_def:
        # For enum fields, use radiolist
        enum_values = field_def["enum"]
        enum_names = field_def.get("enumNames", enum_values)

        values = [(val, name) for val, name in zip(enum_values, enum_names)]

        result = await radiolist_dialog(
            title=title,
            text=prompt_text,
            values=values,
            style=ELICITATION_STYLE,
        ).run_async()

        return result  # None if cancelled

    else:
        # For other fields, use input dialog
        validator = None

        if field_type in ["number", "integer"]:
            validator = NumberValidator(
                field_type=field_type,
                minimum=field_def.get("minimum"),
                maximum=field_def.get("maximum"),
            )
        elif field_type == "string":
            validator = StringValidator(
                min_length=field_def.get("minLength"),
                max_length=field_def.get("maxLength"),
            )

        result = await input_dialog(
            title=title,
            text=prompt_text,
            validator=validator,
            style=ELICITATION_STYLE,
        ).run_async()

        if result is None:  # User cancelled
            return None

        # Convert to appropriate type
        if field_type == "integer":
            return int(result) if result else None
        elif field_type == "number":
            return float(result) if result else None
        else:
            return result


async def build_elicitation_form(
    schema: ElicitRequestedSchema, message: str, agent_name: str, server_name: str
) -> Optional[Dict[str, Any]]:
    """Build and display a form based on the elicitation schema.

    Returns:
        Dict of field values if accepted, None if cancelled
    """

    if schema.get("type") != "object":
        raise ValueError("Schema must have type 'object'")

    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])

    # Show initial information
    from rich.panel import Panel
    from rich.table import Table
    from mcp_agent.console import console

    # Create info table
    info_table = Table(show_header=False, padding=0, box=None)
    info_table.add_column("Label", style="bold yellow")
    info_table.add_column("Value", style="white")
    info_table.add_row("Agent:", agent_name)
    info_table.add_row("Server:", server_name)

    # Create fields summary
    fields_info = []
    for field_name, field_def in properties.items():
        field_type = field_def.get("type", "string")
        title = field_def.get("title", field_name)
        is_required = field_name in required_fields
        required_text = " (required)" if is_required else " (optional)"

        # Add type-specific info
        type_info = field_type
        if field_type == "string" and "enum" in field_def:
            type_info = f"choice from: {', '.join(field_def['enum'])}"
        elif field_type in ["number", "integer"]:
            constraints = []
            if "minimum" in field_def:
                constraints.append(f"min: {field_def['minimum']}")
            if "maximum" in field_def:
                constraints.append(f"max: {field_def['maximum']}")
            if constraints:
                type_info += f" ({', '.join(constraints)})"

        fields_info.append(f"• {title}: {type_info}{required_text}")

    # Create panel content
    from rich.console import Group

    panel_content = Group(
        info_table, "", f"[bold]{message}[/bold]", "", "Required Information:", *fields_info
    )

    panel = Panel(
        panel_content,
        title="[red bold]ELICITATION REQUEST[/red bold]",
        title_align="center",
        style="red",
        border_style="bold red",
        padding=(1, 2),
    )

    console.print(panel)

    # Collect form data
    form_data = {}

    for field_name, field_def in properties.items():
        value = await build_field_input(field_name, field_def, required_fields)

        if value is None:  # User cancelled
            # Ask if they want to cancel the entire form
            cancel_choice = await button_dialog(
                title="Cancel Form?",
                text="Do you want to cancel the entire form or continue with other fields?",
                buttons=[
                    ("continue", "Continue"),
                    ("cancel", "Cancel Form"),
                ],
                style=ELICITATION_STYLE,
            ).run_async()

            if cancel_choice == "cancel":
                return None
            # If continue, treat this field as not provided
            if field_name in required_fields:
                console.print(
                    f"[yellow]Warning: {field_name} is required but was not provided[/yellow]"
                )
        else:
            form_data[field_name] = value

    # Validate required fields
    missing_required = [f for f in required_fields if f not in form_data or form_data[f] is None]
    if missing_required:
        console.print(f"[red]Missing required fields: {', '.join(missing_required)}[/red]")
        retry = await yes_no_dialog(
            title="Missing Required Fields",
            text=f"The following required fields are missing: {', '.join(missing_required)}\n\nDo you want to go back and fill them?",
            yes_text="Go Back",
            no_text="Cancel Form",
            style=ELICITATION_STYLE,
        ).run_async()

        if retry:
            # In a real implementation, we'd loop back to collect missing fields
            # For now, just cancel
            return None
        else:
            return None

    return form_data
