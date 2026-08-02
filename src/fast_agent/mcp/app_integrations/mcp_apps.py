from typing import Any

MCP_APPS_MIME_TYPE = "text/html;profile=mcp-app"
MCP_APPS_RESOURCE_URI_KEY = "ui/resourceUri"
UI_METADATA_KEY = "ui"
UI_RESOURCE_URI_KEY = "resourceUri"
UI_VISIBILITY_KEY = "visibility"


def ui_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    ui = meta.get(UI_METADATA_KEY)
    return ui if isinstance(ui, dict) else {}


def resource_uri(meta: dict[str, Any]) -> str | None:
    nested_value = ui_metadata(meta).get(UI_RESOURCE_URI_KEY)
    if isinstance(nested_value, str) and nested_value:
        return nested_value
    flat_value = meta.get(MCP_APPS_RESOURCE_URI_KEY)
    return flat_value if isinstance(flat_value, str) and flat_value else None


def mark_tool_metadata(
    meta: dict[str, Any],
    resource_uri: str,
    visibility: list[str],
) -> None:
    ui = dict(ui_metadata(meta))
    ui[UI_RESOURCE_URI_KEY] = resource_uri
    ui[UI_VISIBILITY_KEY] = visibility
    meta[UI_METADATA_KEY] = ui
