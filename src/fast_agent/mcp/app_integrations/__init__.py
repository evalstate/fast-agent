from dataclasses import dataclass
from typing import Any

from pydantic import AnyUrl

from . import mcp_apps, openai_apps_sdk
from .models import (
    APP_ONLY_VISIBILITY,
    DEFAULT_APP_VISIBILITY,
    AppIntegrationKind,
    AppResourceConfig,
    AppServerConfig,
    AppToolConfig,
    AppToolMetadata,
    AppVisibility,
)

_VALID_APP_VISIBILITY = frozenset(DEFAULT_APP_VISIBILITY)
APP_INTEGRATION_KIND_KEY = "fast-agent/appIntegrationKind"
APP_INTEGRATION_RESOURCE_URI_KEY = "fast-agent/appResourceUri"


@dataclass(frozen=True, slots=True)
class _VisibilityMetadata:
    values: list[AppVisibility]
    warnings: list[str]


def _app_visibility_value(value: str) -> AppVisibility | None:
    if value == "model":
        return "model"
    if value == "app":
        return "app"
    return None


def _visibility(meta: dict[str, Any]) -> _VisibilityMetadata:
    raw_visibility = mcp_apps.ui_metadata(meta).get(mcp_apps.UI_VISIBILITY_KEY)
    if raw_visibility is None:
        return _VisibilityMetadata(values=list(DEFAULT_APP_VISIBILITY), warnings=[])
    if not isinstance(raw_visibility, list) or not all(
        isinstance(value, str) for value in raw_visibility
    ):
        return _VisibilityMetadata(
            values=list(DEFAULT_APP_VISIBILITY),
            warnings=["invalid _meta.ui.visibility; expected list[str]"],
        )

    visibility = [
        visibility_value
        for value in raw_visibility
        if (visibility_value := _app_visibility_value(value)) is not None
    ]
    invalid = sorted(value for value in raw_visibility if value not in _VALID_APP_VISIBILITY)
    warnings = (
        [f"invalid _meta.ui.visibility values ignored: {', '.join(invalid)}"] if invalid else []
    )
    return _VisibilityMetadata(
        values=visibility or list(DEFAULT_APP_VISIBILITY),
        warnings=warnings,
    )


def extract_app_tool_metadata(
    meta: dict[str, Any], *, namespaced_tool_name: str
) -> AppToolMetadata | None:
    """Normalize OpenAI Apps SDK or MCP Apps metadata from a tool."""
    resource_value = mcp_apps.resource_uri(meta)
    kind = AppIntegrationKind.MCP_APPS
    if resource_value is None:
        resource_value = openai_apps_sdk.resource_uri(meta)
        kind = AppIntegrationKind.OPENAI_APPS_SDK
    if resource_value is None:
        return None

    try:
        resource_uri = AnyUrl(resource_value)
    except Exception as exc:
        raise ValueError(
            f"Tool '{namespaced_tool_name}' resource URI '{resource_value}' is invalid: {exc}"
        ) from exc

    visibility = _visibility(meta)
    return AppToolMetadata(
        resource_uri=resource_uri,
        kind=kind,
        visibility=visibility.values,
        warnings=visibility.warnings,
    )


def integration_kind_for_mime_type(mime_type: str | None) -> AppIntegrationKind | None:
    if mime_type == openai_apps_sdk.OPENAI_APPS_SDK_MIME_TYPE:
        return AppIntegrationKind.OPENAI_APPS_SDK
    if mime_type == mcp_apps.MCP_APPS_MIME_TYPE:
        return AppIntegrationKind.MCP_APPS
    return None


def expected_mime_type(kind: AppIntegrationKind) -> str:
    if kind is AppIntegrationKind.MCP_APPS:
        return mcp_apps.MCP_APPS_MIME_TYPE
    return openai_apps_sdk.OPENAI_APPS_SDK_MIME_TYPE


def supported_mime_types() -> tuple[str, str]:
    return (
        openai_apps_sdk.OPENAI_APPS_SDK_MIME_TYPE,
        mcp_apps.MCP_APPS_MIME_TYPE,
    )


def mark_tool_metadata(meta: dict[str, Any], tool: AppToolConfig) -> None:
    if tool.resource_uri is None or tool.kind is None:
        return
    resource_uri = str(tool.resource_uri)
    if tool.kind is AppIntegrationKind.MCP_APPS:
        mcp_apps.mark_tool_metadata(meta, resource_uri, list(tool.visibility))
    meta[APP_INTEGRATION_KIND_KEY] = tool.kind.value
    meta[APP_INTEGRATION_RESOURCE_URI_KEY] = resource_uri


__all__ = [
    "APP_ONLY_VISIBILITY",
    "APP_INTEGRATION_KIND_KEY",
    "APP_INTEGRATION_RESOURCE_URI_KEY",
    "DEFAULT_APP_VISIBILITY",
    "AppIntegrationKind",
    "AppResourceConfig",
    "AppServerConfig",
    "AppToolConfig",
    "AppToolMetadata",
    "AppVisibility",
    "expected_mime_type",
    "extract_app_tool_metadata",
    "integration_kind_for_mime_type",
    "mark_tool_metadata",
    "supported_mime_types",
]
