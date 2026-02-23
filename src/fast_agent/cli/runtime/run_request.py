"""Typed request model for CLI runtime execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict

if TYPE_CHECKING:
    from pathlib import Path

Mode = Literal["interactive", "serve"]


class UrlServerConfig(TypedDict, total=False):
    """Configuration for URL-backed MCP servers."""

    transport: str
    url: str
    headers: dict[str, str]
    auth: dict[str, str | bool]


class StdioServerConfig(TypedDict):
    """Configuration for STDIO-backed MCP servers."""

    transport: Literal["stdio"]
    command: str
    args: list[str]


@dataclass(slots=True)
class AgentRunRequest:
    """Normalized request used by the CLI runtime."""

    name: str
    instruction: str | None
    config_path: str | None
    server_list: list[str] | None
    agent_cards: list[str] | None
    card_tools: list[str] | None
    model: str | None
    message: str | None
    prompt_file: str | None
    result_file: str | None
    resume: str | None
    url_servers: dict[str, UrlServerConfig] | None
    stdio_servers: dict[str, StdioServerConfig] | None
    agent_name: str | None
    target_agent_name: str | None
    skills_directory: Path | None
    environment_dir: Path | None
    noenv: bool
    force_smart: bool
    shell_runtime: bool
    mode: Mode
    transport: str
    host: str
    port: int
    tool_description: str | None
    tool_name_template: str | None
    instance_scope: str
    permissions_enabled: bool
    reload: bool
    watch: bool
    missing_shell_cwd_policy: Literal["ask", "create", "warn", "error"] | None = None

    def __post_init__(self) -> None:
        if self.noenv and self.environment_dir is not None:
            raise ValueError("--noenv cannot be combined with --env")

    @property
    def allow_sessions(self) -> bool:
        return not self.noenv

    @property
    def allow_implicit_cards(self) -> bool:
        return not self.noenv

    @property
    def allow_permission_store(self) -> bool:
        return not self.noenv

    def to_agent_setup_kwargs(self) -> dict[str, Any]:
        """Convert to the legacy kwargs shape used by `_run_agent` wrappers."""
        return {
            "name": self.name,
            "instruction": self.instruction,
            "config_path": self.config_path,
            "server_list": self.server_list,
            "agent_cards": self.agent_cards,
            "card_tools": self.card_tools,
            "model": self.model,
            "message": self.message,
            "prompt_file": self.prompt_file,
            "result_file": self.result_file,
            "resume": self.resume,
            "url_servers": self.url_servers,
            "stdio_servers": self.stdio_servers,
            "agent_name": self.agent_name,
            "target_agent_name": self.target_agent_name,
            "skills_directory": self.skills_directory,
            "environment_dir": self.environment_dir,
            "noenv": self.noenv,
            "force_smart": self.force_smart,
            "shell_runtime": self.shell_runtime,
            "mode": self.mode,
            "transport": self.transport,
            "host": self.host,
            "port": self.port,
            "tool_description": self.tool_description,
            "tool_name_template": self.tool_name_template,
            "instance_scope": self.instance_scope,
            "permissions_enabled": self.permissions_enabled,
            "reload": self.reload,
            "watch": self.watch,
            "missing_shell_cwd_policy": self.missing_shell_cwd_policy,
        }
