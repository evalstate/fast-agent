from __future__ import annotations

import asyncio
from typing import Any, cast

from fast_agent.human_input.form_fields import FormSchema, string
from fast_agent.human_input.simple_form import form


def test_simple_form_sets_title_and_hides_server_name(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def _fake_show_simple_elicitation_form(*, schema, message, agent_name, server_name):
        captured["schema"] = schema
        captured["message"] = message
        captured["agent_name"] = agent_name
        captured["server_name"] = server_name
        return "accept", {"name": "Ada"}

    monkeypatch.setattr(
        "fast_agent.ui.elicitation_form.show_simple_elicitation_form",
        _fake_show_simple_elicitation_form,
    )

    result = asyncio.run(
        form(
            FormSchema(name=string(title="Name", description="Your name")).required("name"),
            message="Editing: /tmp/fastagent.config.yaml",
            title="Display Settings",
        )
    )

    assert result == {"name": "Ada"}
    assert captured["agent_name"] == "Display Settings"
    assert captured["server_name"] == ""
    schema = cast("dict[str, Any]", captured["schema"])
    assert schema["title"] == "Display Settings"
