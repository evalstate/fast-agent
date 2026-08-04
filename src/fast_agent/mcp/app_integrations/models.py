from enum import StrEnum
from typing import Any, Literal

from pydantic import AnyUrl, BaseModel, Field

type AppVisibility = Literal["model", "app"]
DEFAULT_APP_VISIBILITY: tuple[AppVisibility, ...] = ("model", "app")
APP_ONLY_VISIBILITY: frozenset[AppVisibility] = frozenset(("app",))


class AppIntegrationKind(StrEnum):
    """Interactive app protocols discovered from MCP metadata."""

    OPENAI_APPS_SDK = "openai_apps_sdk"
    MCP_APPS = "mcp_apps"

    @property
    def display_name(self) -> str:
        if self is AppIntegrationKind.MCP_APPS:
            return "MCP Apps"
        return "OpenAI Apps SDK"


class AppToolMetadata(BaseModel):
    """Normalized app metadata extracted from an MCP tool."""

    resource_uri: AnyUrl
    kind: AppIntegrationKind
    visibility: list[AppVisibility] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def is_app_only(self) -> bool:
        return set(self.visibility) == APP_ONLY_VISIBILITY


class AppResourceConfig(BaseModel):
    """Interactive app resource exposed by an MCP server."""

    uri: AnyUrl
    mime_type: str | None = None
    kind: AppIntegrationKind | None = None
    warning: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.kind is not None


class AppToolConfig(BaseModel):
    """Interactive app metadata discovered for an MCP tool."""

    tool_name: str
    namespaced_tool_name: str
    resource_uri: AnyUrl | None = None
    linked_resource_uri: AnyUrl | None = None
    kind: AppIntegrationKind | None = None
    visibility: list[AppVisibility] = Field(default_factory=list)
    warning: str | None = None

    @property
    def display_name(self) -> str:
        return self.namespaced_tool_name or self.tool_name

    @property
    def is_app_only(self) -> bool:
        return set(self.visibility) == APP_ONLY_VISIBILITY

    @property
    def is_valid(self) -> bool:
        return (
            self.kind is not None and self.linked_resource_uri is not None and self.warning is None
        )


class AppServerConfig(BaseModel):
    """Interactive app configuration discovered for an MCP server."""

    server_name: str
    supports_resources: bool = False
    resources: list[AppResourceConfig] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    tools: list[AppToolConfig] = Field(default_factory=list)

    @property
    def enabled(self) -> bool:
        return any(resource.is_valid for resource in self.resources)
