import json
from typing import Any

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


def _effective_agent_name(
    request: HumanInputRequest,
    agent_name: str | None,
) -> str:
    if agent_name:
        return agent_name
    if request.metadata:
        return request.metadata.get("agent_name", "Unknown Agent")
    return "Unknown Agent"


def _requested_schema(request: HumanInputRequest) -> dict[str, Any] | None:
    if not request.metadata:
        return None
    schema = request.metadata.get("requested_schema")
    return schema if isinstance(schema, dict) else None


def _start_elicitation_tracking(server_name: str) -> None:
    elicitation_state.start_elicitation(server_name)
    try:
        from fast_agent.ui import notification_tracker

        notification_tracker.start_elicitation(server_name)
    except Exception:
        # Don't let notification tracking break elicitation
        pass


def _end_elicitation_tracking(server_name: str) -> None:
    elicitation_state.end_elicitation(server_name)
    try:
        from fast_agent.ui import notification_tracker

        notification_tracker.end_elicitation(server_name)
    except Exception:
        # Don't let notification tracking break elicitation
        pass


def _cancelled_response(request_id: str) -> HumanInputResponse:
    return HumanInputResponse(
        request_id=request_id,
        response="__CANCELLED__",
        metadata={"auto_cancelled": True, "reason": "Server elicitation disabled by user"},
    )


async def _prompt_for_schema(
    *,
    schema: dict[str, Any],
    request: HumanInputRequest,
    agent_name: str,
    server_name: str,
) -> str:
    form_action, form_data = await show_simple_elicitation_form(
        schema=schema,
        message=request.prompt,
        agent_name=agent_name,
        server_name=server_name,
    )

    if form_action == "accept" and form_data is not None:
        return json.dumps(form_data)
    if form_action == "decline":
        return "__DECLINED__"
    if form_action == "disable":
        return "__DISABLE_SERVER__"
    return "__CANCELLED__"


async def _prompt_for_text(
    *,
    request: HumanInputRequest,
    agent_name: str,
    server_name: str,
) -> str:
    from prompt_toolkit.shortcuts import input_dialog

    response = await input_dialog(
        title="Input Requested",
        text=f"Agent: {agent_name}\nServer: {server_name}\n\n{request.prompt}",
        style=ELICITATION_STYLE,
    ).run_async()
    return response if response is not None else "__CANCELLED__"


async def _prompt_for_elicitation(
    *,
    request: HumanInputRequest,
    agent_name: str,
    server_name: str,
    schema: dict[str, Any] | None,
) -> str:
    with progress_display.paused():
        try:
            if schema:
                return await _prompt_for_schema(
                    schema=schema,
                    request=request,
                    agent_name=agent_name,
                    server_name=server_name,
                )
            return await _prompt_for_text(
                request=request,
                agent_name=agent_name,
                server_name=server_name,
            )
        except (KeyboardInterrupt, EOFError):
            return "__CANCELLED__"


async def elicitation_input_callback(
    request: HumanInputRequest,
    agent_name: str | None = None,
    server_name: str | None = None,
    server_info: dict[str, Any] | None = None,
) -> HumanInputResponse:
    """Request input from a human user for MCP server elicitation requests."""
    del server_info

    effective_agent_name = _effective_agent_name(request, agent_name)
    effective_server_name = server_name or "Unknown Server"

    _start_elicitation_tracking(effective_server_name)

    try:
        request_id = request.request_id or ""
        if elicitation_state.is_disabled(effective_server_name):
            return _cancelled_response(request_id)

        schema = _requested_schema(request)
        response = await _prompt_for_elicitation(
            request=request,
            agent_name=effective_agent_name,
            server_name=effective_server_name,
            schema=schema,
        )
        return HumanInputResponse(
            request_id=request_id,
            response=response.strip(),
            metadata={"has_schema": schema is not None},
        )
    finally:
        _end_elicitation_tracking(effective_server_name)


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
