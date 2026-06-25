from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from pydantic import AnyUrl, BaseModel, Field

SKYBRIDGE_MIME_TYPE = "text/html+skybridge"
MCP_APP_MIME_TYPE = "text/html;profile=mcp-app"

OPENAI_OUTPUT_TEMPLATE_KEY = "openai/outputTemplate"
MCP_APP_RESOURCE_URI_KEY = "ui/resourceUri"
type AppVisibility = Literal["model", "app"]
DEFAULT_APP_VISIBILITY: tuple[AppVisibility, ...] = ("model", "app")
APP_ONLY_VISIBILITY: frozenset[AppVisibility] = frozenset(("app",))
_VALID_APP_VISIBILITY = frozenset(DEFAULT_APP_VISIBILITY)


class AppIntegrationKind(StrEnum):
    """Interactive UI integration variants discovered from MCP metadata."""

    SKYBRIDGE = "skybridge"
    MCP_APP = "mcp_app"

    @property
    def display_name(self) -> str:
        if self is AppIntegrationKind.MCP_APP:
            return "MCP Apps"
        return "OpenAI Apps SDK"

    @property
    def expected_mime_type(self) -> str:
        if self is AppIntegrationKind.MCP_APP:
            return MCP_APP_MIME_TYPE
        return SKYBRIDGE_MIME_TYPE


class AppToolMetadata(BaseModel):
    """Normalized app metadata extracted from a tool."""

    resource_uri: AnyUrl
    kind: AppIntegrationKind
    visibility: list[AppVisibility] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def is_app_only(self) -> bool:
        return set(self.visibility) == APP_ONLY_VISIBILITY


@dataclass(frozen=True, slots=True)
class VisibilityMetadata:
    values: list[AppVisibility]
    warnings: list[str]


def _ui_meta(meta: dict[str, Any]) -> dict[str, Any]:
    ui = meta.get("ui")
    if isinstance(ui, dict):
        return ui
    return {}


def _app_visibility_value(value: str) -> AppVisibility | None:
    if value == "model":
        return "model"
    if value == "app":
        return "app"
    return None


def _visibility(meta: dict[str, Any]) -> VisibilityMetadata:
    ui = _ui_meta(meta)
    raw_visibility = ui.get("visibility")
    if raw_visibility is None:
        return VisibilityMetadata(values=list(DEFAULT_APP_VISIBILITY), warnings=[])
    if not isinstance(raw_visibility, list) or not all(
        isinstance(value, str) for value in raw_visibility
    ):
        return VisibilityMetadata(
            values=list(DEFAULT_APP_VISIBILITY),
            warnings=["invalid _meta.ui.visibility; expected list[str]"],
        )

    visibility = [
        visibility_value
        for value in raw_visibility
        if (visibility_value := _app_visibility_value(value)) is not None
    ]
    invalid = sorted(set(raw_visibility) - _VALID_APP_VISIBILITY)
    warnings = (
        [f"invalid _meta.ui.visibility values ignored: {', '.join(invalid)}"] if invalid else []
    )
    return VisibilityMetadata(
        values=visibility or list(DEFAULT_APP_VISIBILITY),
        warnings=warnings,
    )


def extract_app_tool_metadata(
    meta: dict[str, Any], *, namespaced_tool_name: str
) -> AppToolMetadata | None:
    """Return normalized app metadata for either MCP Apps or Skybridge tools."""

    warnings: list[str] = []
    ui = _ui_meta(meta)
    resource_value = ui.get("resourceUri")
    kind = AppIntegrationKind.MCP_APP

    if not isinstance(resource_value, str) or not resource_value:
        resource_value = meta.get(MCP_APP_RESOURCE_URI_KEY)

    if not isinstance(resource_value, str) or not resource_value:
        resource_value = meta.get(OPENAI_OUTPUT_TEMPLATE_KEY)
        kind = AppIntegrationKind.SKYBRIDGE

    if not isinstance(resource_value, str) or not resource_value:
        return None

    try:
        resource_uri = AnyUrl(resource_value)
    except Exception as exc:
        raise ValueError(
            f"Tool '{namespaced_tool_name}' resource URI '{resource_value}' is invalid: {exc}"
        ) from exc

    visibility = _visibility(meta)
    warnings.extend(visibility.warnings)

    return AppToolMetadata(
        resource_uri=resource_uri,
        kind=kind,
        visibility=visibility.values,
        warnings=warnings,
    )


class SkybridgeResourceConfig(BaseModel):
    """Represents an interactive app resource exposed by an MCP server."""

    uri: AnyUrl
    mime_type: str | None = None
    kind: AppIntegrationKind | None = None
    is_skybridge: bool = False
    is_mcp_app: bool = False
    warning: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid_app_resource(self) -> bool:
        return self.is_skybridge or self.is_mcp_app


class SkybridgeToolConfig(BaseModel):
    """Represents interactive app metadata discovered for a tool."""

    tool_name: str
    namespaced_tool_name: str
    template_uri: AnyUrl | None = None
    resource_uri: AnyUrl | None = None
    kind: AppIntegrationKind = AppIntegrationKind.SKYBRIDGE
    visibility: list[AppVisibility] = Field(default_factory=list)
    is_valid: bool = False
    warning: str | None = None

    @property
    def display_name(self) -> str:
        return self.namespaced_tool_name or self.tool_name

    @property
    def is_app_only(self) -> bool:
        return set(self.visibility) == APP_ONLY_VISIBILITY


class SkybridgeServerConfig(BaseModel):
    """Interactive app configuration discovered for a specific MCP server."""

    server_name: str
    supports_resources: bool = False
    ui_resources: list[SkybridgeResourceConfig] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    tools: list[SkybridgeToolConfig] = Field(default_factory=list)

    @property
    def enabled(self) -> bool:
        """Return True when at least one resource advertises a supported app MIME type."""
        return any(resource.is_valid_app_resource for resource in self.ui_resources)
