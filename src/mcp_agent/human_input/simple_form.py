"""Simple form API for elicitation schemas without MCP wrappers."""

import asyncio
from typing import Any, Dict, Optional, Union

from mcp.types import ElicitRequestedSchema

from mcp_agent.human_input.elicitation_form import show_simple_elicitation_form
from mcp_agent.human_input.form_fields import FormSchema


async def form(
    schema: Union[FormSchema, ElicitRequestedSchema, Dict[str, Any]],
    message: str = "Please fill out the form",
    title: str = "Form Input",
) -> Optional[Dict[str, Any]]:
    """
    Simple form API that presents an elicitation form and returns results.

    Args:
        schema: FormSchema, ElicitRequestedSchema, or dict schema
        message: Message to display to the user
        title: Title for the form (used as agent_name)

    Returns:
        Dict with form data if accepted, None if cancelled/declined

    Example:
        from mcp_agent.human_input.form_fields import FormSchema, string, email, integer

        schema = FormSchema(
            name=string("Name", "Your full name", min_length=2),
            email=email("Email", "Your email address"),
            age=integer("Age", "Your age", minimum=0, maximum=120)
        ).required("name", "email")

        result = await form(schema, "Please enter your information")
        if result:
            print(f"Name: {result['name']}, Email: {result['email']}")
    """
    # Convert schema to ElicitRequestedSchema format
    if isinstance(schema, FormSchema):
        elicit_schema = schema.to_schema()
    elif isinstance(schema, dict):
        elicit_schema = schema
    else:
        elicit_schema = schema

    # Show the form
    action, result = await show_simple_elicitation_form(
        schema=elicit_schema, message=message, agent_name=title, server_name="SimpleForm"
    )

    # Return results based on action
    if action == "accept":
        return result
    else:
        return None


def form_sync(
    schema: Union[FormSchema, ElicitRequestedSchema, Dict[str, Any]],
    message: str = "Please fill out the form",
    title: str = "Form Input",
) -> Optional[Dict[str, Any]]:
    """
    Synchronous wrapper for the form function.

    Args:
        schema: FormSchema, ElicitRequestedSchema, or dict schema
        message: Message to display to the user
        title: Title for the form (used as agent_name)

    Returns:
        Dict with form data if accepted, None if cancelled/declined
    """
    return asyncio.run(form(schema, message, title))


# Convenience function with a shorter name
async def ask(
    schema: Union[FormSchema, ElicitRequestedSchema, Dict[str, Any]],
    message: str = "Please provide the requested information",
) -> Optional[Dict[str, Any]]:
    """
    Short alias for form() function.

    Example:
        from mcp_agent.human_input.form_fields import FormSchema, string, email

        schema = FormSchema(
            name=string("Name", "Your name"),
            email=email("Email", "Your email")
        ).required("name")

        result = await ask(schema, "What's your info?")
    """
    return await form(schema, message)


def ask_sync(
    schema: Union[FormSchema, ElicitRequestedSchema, Dict[str, Any]],
    message: str = "Please provide the requested information",
) -> Optional[Dict[str, Any]]:
    """
    Synchronous version of ask().

    Example:
        result = ask_sync(schema, "What's your info?")
    """
    return form_sync(schema, message)
