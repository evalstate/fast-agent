from typing import Any, Dict, Optional

from mcp_agent.human_input.elicitation_form import (
    show_simple_elicitation_form,
)
from mcp_agent.human_input.elicitation_forms import (
    ELICITATION_STYLE,
)
from mcp_agent.human_input.elicitation_state import elicitation_state
from mcp_agent.human_input.types import (
    HumanInputRequest,
    HumanInputResponse,
)
from mcp_agent.progress_display import progress_display


async def elicitation_input_callback(
    request: HumanInputRequest,
    agent_name: str | None = None,
    server_name: str | None = None,
    server_info: dict[str, Any] | None = None,
) -> HumanInputResponse:
    """Request input from a human user for MCP server elicitation requests."""

    # Extract effective names
    effective_agent_name = agent_name or (
        request.metadata.get("agent_name", "Unknown Agent") if request.metadata else "Unknown Agent"
    )
    effective_server_name = server_name or "Unknown Server"

    # Check if elicitation is disabled for this server
    if elicitation_state.is_disabled(effective_server_name):
        return HumanInputResponse(
            request_id=request.request_id,
            response="__CANCELLED__",
            metadata={"auto_cancelled": True, "reason": "Server elicitation disabled by user"},
        )

    # Get the elicitation schema from metadata
    schema: Optional[Dict[str, Any]] = None
    if request.metadata and "requested_schema" in request.metadata:
        schema = request.metadata["requested_schema"]

    # Use the context manager to pause the progress display while getting input
    with progress_display.paused():
        try:
            if schema:
                form_action, form_data = await show_simple_elicitation_form(
                    schema=schema,
                    message=request.prompt,
                    agent_name=effective_agent_name,
                    server_name=effective_server_name,
                )

                if form_action == "accept" and form_data is not None:
                    # Convert form data to JSON string
                    import json

                    response = json.dumps(form_data)
                elif form_action == "decline":
                    response = "__DECLINED__"
                elif form_action == "disable":
                    response = "__DISABLE_SERVER__"
                else:  # cancel
                    response = "__CANCELLED__"
            else:
                # No schema, fall back to text input using prompt_toolkit only
                from prompt_toolkit.shortcuts import input_dialog

                response = await input_dialog(
                    title="Input Requested",
                    text=f"Agent: {effective_agent_name}\nServer: {effective_server_name}\n\n{request.prompt}",
                    style=ELICITATION_STYLE,
                ).run_async()

                if response is None:
                    response = "__CANCELLED__"

        except KeyboardInterrupt:
            response = "__CANCELLED__"
        except EOFError:
            response = "__CANCELLED__"

    return HumanInputResponse(
        request_id=request.request_id,
        response=response.strip() if isinstance(response, str) else response,
        metadata={"has_schema": schema is not None},
    )
