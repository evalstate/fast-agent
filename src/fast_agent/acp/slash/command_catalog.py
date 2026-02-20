"""Slash command catalog helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from acp.schema import AvailableCommandInput, UnstructuredCommandInput

if TYPE_CHECKING:
    from acp.schema import AvailableCommand


def apply_dynamic_session_hints(
    commands: dict[str, "AvailableCommand"], model_hint: str
) -> list["AvailableCommand"]:
    model_command = commands.get("model")
    if model_command:
        commands["model"] = model_command.model_copy(
            update={
                "input": AvailableCommandInput(root=UnstructuredCommandInput(hint=model_hint)),
            }
        )
    return list(commands.values())
