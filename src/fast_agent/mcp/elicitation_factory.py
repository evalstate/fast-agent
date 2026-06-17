"""
Factory for resolving elicitation handlers with proper precedence.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeGuard, cast

from mcp.client.session import ElicitationFnT

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.config import MCPElicitationSettings, MCPServerSettings, Settings
from fast_agent.core.logging.logger import get_logger
from fast_agent.mcp.elicitation_handlers import (
    auto_cancel_elicitation_handler,
    forms_elicitation_handler,
)
from fast_agent.utils.text import strip_casefold

logger = get_logger(__name__)

ElicitationMode = Literal["forms", "auto-cancel", "none"]
type ElicitationConfigContainer = Settings | MCPServerSettings | Mapping[str, object] | None
type ElicitationConfig = MCPElicitationSettings | Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class _ElicitationModeResolution:
    handler: ElicitationFnT | None
    log_message: str


_ELICITATION_MODE_RESOLUTIONS: Mapping[str, _ElicitationModeResolution] = {
    "none": _ElicitationModeResolution(
        handler=None,
        log_message="Elicitation disabled by {source} config for agent {agent_name}",
    ),
    "auto-cancel": _ElicitationModeResolution(
        handler=auto_cancel_elicitation_handler,
        log_message=(
            "Using auto-cancel elicitation handler ({source} config) for agent {agent_name}"
        ),
    ),
    "forms": _ElicitationModeResolution(
        handler=forms_elicitation_handler,
        log_message="Using forms elicitation handler ({source} config) for agent {agent_name}",
    ),
}


def _elicitation_config_from_container(container: ElicitationConfigContainer) -> ElicitationConfig:
    if container is None:
        return None
    if isinstance(container, Mapping):
        config_map = cast("Mapping[str, object]", container)
        elicitation_config = config_map.get("elicitation")
        return elicitation_config if _is_elicitation_config(elicitation_config) else None
    if isinstance(container, MCPServerSettings):
        return container.elicitation
    extra = container.model_extra or {}
    elicitation_config = extra.get("elicitation")
    return elicitation_config if _is_elicitation_config(elicitation_config) else None


def _is_elicitation_config(value: object) -> TypeGuard[ElicitationConfig]:
    return value is None or isinstance(value, MCPElicitationSettings | Mapping)


def _elicitation_mode_from_config(
    elicitation_config: ElicitationConfig,
    *,
    default: ElicitationMode | None,
) -> str | None:
    if elicitation_config is None:
        return default
    if isinstance(elicitation_config, Mapping):
        config_map = cast("Mapping[str, object]", elicitation_config)
        raw_mode = config_map.get("mode")
        return cast("str | None", raw_mode if raw_mode is not None else default)
    return elicitation_config.mode or default


def _handler_for_elicitation_mode(
    mode: str | None,
    *,
    source: str,
    agent_name: str,
) -> ElicitationFnT | None:
    mode_key = mode if mode in _ELICITATION_MODE_RESOLUTIONS else "forms"
    resolution = _ELICITATION_MODE_RESOLUTIONS[mode_key]
    logger.debug(resolution.log_message.format(source=source, agent_name=agent_name))
    return resolution.handler


def resolve_elicitation_handler(
    agent_config: AgentConfig,
    app_config: ElicitationConfigContainer,
    server_config: ElicitationConfigContainer = None,
) -> ElicitationFnT | None:
    """Resolve elicitation handler with proper precedence.

    Precedence order:
    1. Agent decorator supplied (highest precedence)
    2. Server-specific config file setting
    3. Global config file setting
    4. Default forms handler (lowest precedence)

    Args:
        agent_config: Agent configuration from decorator
        app_config: Application configuration from YAML
        server_config: Server-specific configuration (optional)

    Returns:
        ElicitationFnT handler or None (no elicitation capability)
    """

    # 1. Decorator takes highest precedence
    if agent_config.elicitation_handler:
        logger.debug(f"Using decorator-provided elicitation handler for agent {agent_config.name}")
        return agent_config.elicitation_handler

    # 2. Check server-specific config first
    if server_config:
        mode = _elicitation_mode_from_config(
            _elicitation_config_from_container(server_config),
            default=None,
        )
        if mode:
            return _handler_for_elicitation_mode(
                mode,
                source="server",
                agent_name=agent_config.name,
            )

    # 3. Check global config file
    mode = _elicitation_mode_from_config(
        _elicitation_config_from_container(app_config),
        default="forms",
    )
    return _handler_for_elicitation_mode(
        mode,
        source="global",
        agent_name=agent_config.name,
    )


def resolve_global_elicitation_mode(app_config: ElicitationConfigContainer) -> str | None:
    """Resolve only the global elicitation mode used for session status display."""
    mode = _elicitation_mode_from_config(
        _elicitation_config_from_container(app_config),
        default=None,
    )
    return strip_casefold(mode) if isinstance(mode, str) else None
