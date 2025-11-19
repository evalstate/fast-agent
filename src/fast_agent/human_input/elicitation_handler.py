from typing import Any, Dict, Optional

from fast_agent.human_input.elicitation_state import elicitation_state
from fast_agent.human_input.types import (
    HumanInputRequest,
    HumanInputResponse,
)
from fast_agent.tools.elicitation import set_elicitation_input_callback
from fast_agent.ui.elicitation_form import (
    show_simple_elicitation_form,
)
from fast_agent.ui.elicitation_style import (
    ELICITATION_STYLE,
)
from fast_agent.ui.progress_display import progress_display


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

    # Start tracking elicitation operation
    try:
        from fast_agent.ui import notification_tracker
        notification_tracker.start_elicitation(effective_server_name)
    except Exception:
        # Don't let notification tracking break elicitation
        pass

    try:
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
    finally:
        # End tracking elicitation operation
        try:
            from fast_agent.ui import notification_tracker
            notification_tracker.end_elicitation(effective_server_name)
        except Exception:
            # Don't let notification tracking break elicitation
            pass


# Register adapter with fast_agent tools so they can invoke this UI handler without importing types
async def _elicitation_adapter(
    request_payload: dict,
    agent_name: str | None = None,
    server_name: str | None = None,
    server_info: dict[str, Any] | None = None,
) -> str:
    req = HumanInputRequest(**request_payload)
    resp = await elicitation_input_callback(
        request=req, agent_name=agent_name, server_name=server_name, server_info=server_info
    )
    return resp.response if isinstance(resp.response, str) else str(resp.response)


set_elicitation_input_callback(_elicitation_adapter)
