from typing import List

from pydantic import AnyUrl, BaseModel, Field

SKYBRIDGE_MIME_TYPE = "text/html+skybridge"


class SkybridgeResourceConfig(BaseModel):
    """Represents a Skybridge (apps SDK) resource exposed by an MCP server."""

    uri: AnyUrl
    mime_type: str | None = None
    is_skybridge: bool = False
    warning: str | None = None


class SkybridgeToolConfig(BaseModel):
    """Represents Skybridge metadata discovered for a tool."""

    tool_name: str
    namespaced_tool_name: str
    template_uri: AnyUrl | None = None
    resource_uri: AnyUrl | None = None
    is_valid: bool = False
    warning: str | None = None

    @property
    def display_name(self) -> str:
        return self.namespaced_tool_name or self.tool_name


class SkybridgeServerConfig(BaseModel):
    """Skybridge configuration discovered for a specific MCP server."""

    server_name: str
    supports_resources: bool = False
    ui_resources: List[SkybridgeResourceConfig] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    tools: List[SkybridgeToolConfig] = Field(default_factory=list)

    @property
    def enabled(self) -> bool:
        """Return True when at least one resource advertises the Skybridge MIME type."""
        return any(resource.is_skybridge for resource in self.ui_resources)

