from __future__ import annotations

from typing import Literal, TypeGuard

type McpUIMode = Literal["disabled", "enabled", "auto"]
MCP_UI_MODES: tuple[McpUIMode, ...] = ("disabled", "enabled", "auto")


def is_mcp_ui_mode(value: object) -> TypeGuard[McpUIMode]:
    return value in MCP_UI_MODES


def normalize_mcp_ui_mode(value: object) -> McpUIMode:
    if is_mcp_ui_mode(value):
        return value
    return "auto"
