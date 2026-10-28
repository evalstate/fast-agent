from __future__ import annotations

import asyncio
from contextlib import contextmanager

import pytest

from fast_agent.human_input import elicitation_handler
from fast_agent.human_input.types import HumanInputRequest


@pytest.mark.asyncio
async def test_elicitation_reports_blocked_only_while_waiting(monkeypatch) -> None:
    events: list[str] = []

    @contextmanager
    def blocked_scope():
        events.append("blocked")
        try:
            yield
        finally:
            events.append("working")

    async def prompt_for_elicitation(**_kwargs) -> str:
        events.append("waiting")
        return "answer"

    monkeypatch.setattr(elicitation_handler, "herdr_blocked", blocked_scope)
    monkeypatch.setattr(
        elicitation_handler,
        "_prompt_for_elicitation",
        prompt_for_elicitation,
    )

    response = await elicitation_handler.elicitation_input_callback(
        HumanInputRequest(request_id="request-1", prompt="Continue?")
    )

    assert response.response == "answer"
    assert events == ["blocked", "waiting", "working"]


@pytest.mark.asyncio
async def test_disabled_elicitation_does_not_report_blocked(monkeypatch) -> None:
    entered = False

    @contextmanager
    def blocked_scope():
        nonlocal entered
        entered = True
        yield

    monkeypatch.setattr(elicitation_handler, "herdr_blocked", blocked_scope)
    elicitation_handler.elicitation_state.disable_server("server")

    try:
        response = await elicitation_handler.elicitation_input_callback(
            HumanInputRequest(request_id="request-2", prompt="Continue?"),
            server_name="server",
        )
    finally:
        elicitation_handler.elicitation_state.clear_all()

    assert response.response == "__CANCELLED__"
    assert entered is False


@pytest.mark.asyncio
async def test_concurrent_elicitations_prompt_one_at_a_time(monkeypatch) -> None:
    active = 0
    overlapped = False
    prompted: list[str] = []

    async def prompt_for_elicitation(*, request: HumanInputRequest, server_name: str, **_kw) -> str:
        nonlocal active, overlapped
        active += 1
        overlapped |= active > 1
        prompted.append(request.request_id or "")
        await asyncio.sleep(0.01)
        active -= 1
        if request.request_id == "second":
            # The user picks "disable server" in the second form.
            elicitation_handler.elicitation_state.disable_server(server_name)
        return "answer"

    monkeypatch.setattr(elicitation_handler, "_prompt_for_elicitation", prompt_for_elicitation)

    try:
        responses = await asyncio.gather(
            *(
                elicitation_handler.elicitation_input_callback(
                    HumanInputRequest(request_id=request_id, prompt="Continue?"),
                    server_name="server",
                )
                for request_id in ("first", "second", "third")
            )
        )
    finally:
        elicitation_handler.elicitation_state.clear_all()

    assert not overlapped
    assert prompted == ["first", "second"]
    assert [response.response for response in responses] == [
        "answer",
        "answer",
        "__CANCELLED__",
    ]
