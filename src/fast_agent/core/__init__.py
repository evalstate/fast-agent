"""
Core interfaces for fast-agent.

Public API:
- `Core`: The core application container
- `AgentApp`: Container for interacting with agents
- `FastAgent`: High-level, decorator-driven application class
- `DecoratorMixin`: Mixin providing decorator methods (@agent, @router, etc.)
- `DefaultHarnessApp`: Default harness app boundary for UI/protocol adapters

Note: Agent decorators are accessed via FastAgent instances, e.g.:
    fast = FastAgent("my-app")
    @fast.agent(name="my-agent")
    async def main(): ...

Exports are resolved lazily to avoid circular imports during package init.
"""

from typing import TYPE_CHECKING

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentApp": (".agent_app", "AgentApp"),
    "AgentRuntimeEnvironment": (".harness_app", "AgentRuntimeEnvironment"),
    "AppOpenRequest": (".harness_app", "AppOpenRequest"),
    "Core": (".core_app", "Core"),
    "DefaultHarnessApp": (".harness_app", "DefaultHarnessApp"),
    "DefaultHarnessAppSession": (".harness_app", "DefaultHarnessAppSession"),
    "FastAgent": (".fastagent", "FastAgent"),
    "HarnessApp": (".harness_app", "HarnessApp"),
    "HarnessAppContext": (".harness_app", "HarnessAppContext"),
    "HarnessAppFactory": (".harness_app", "HarnessAppFactory"),
    "HarnessAppSession": (".harness_app", "HarnessAppSession"),
    "HarnessSessionProvider": (".harness_app", "HarnessSessionProvider"),
    "HarnessSessionsAppProvider": (".harness_app", "HarnessSessionsAppProvider"),
    "load_harness_app": (".harness_app", "load_harness_app"),
    "DecoratorMixin": (".direct_decorators", "DecoratorMixin"),
    "RuntimeAgent": (".harness_app", "RuntimeAgent"),
    "RuntimeSkills": (".harness_app", "RuntimeSkills"),
    "RuntimeTools": (".harness_app", "RuntimeTools"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from exc
    from importlib import import_module

    return getattr(import_module(module_name, __name__), attr_name)


if TYPE_CHECKING:  # pragma: no cover - typing aid only
    from .agent_app import AgentApp as AgentApp
    from .core_app import Core as Core
    from .direct_decorators import DecoratorMixin as DecoratorMixin
    from .fastagent import FastAgent as FastAgent
    from .harness_app import AgentRuntimeEnvironment as AgentRuntimeEnvironment
    from .harness_app import AppOpenRequest as AppOpenRequest
    from .harness_app import DefaultHarnessApp as DefaultHarnessApp
    from .harness_app import DefaultHarnessAppSession as DefaultHarnessAppSession
    from .harness_app import HarnessApp as HarnessApp
    from .harness_app import HarnessAppContext as HarnessAppContext
    from .harness_app import HarnessAppFactory as HarnessAppFactory
    from .harness_app import HarnessAppSession as HarnessAppSession
    from .harness_app import HarnessSessionProvider as HarnessSessionProvider
    from .harness_app import HarnessSessionsAppProvider as HarnessSessionsAppProvider
    from .harness_app import RuntimeAgent as RuntimeAgent
    from .harness_app import RuntimeSkills as RuntimeSkills
    from .harness_app import RuntimeTools as RuntimeTools
    from .harness_app import load_harness_app as load_harness_app


__all__ = [
    "AgentApp",
    "AgentRuntimeEnvironment",
    "AppOpenRequest",
    "Core",
    "DecoratorMixin",
    "DefaultHarnessApp",
    "DefaultHarnessAppSession",
    "FastAgent",
    "HarnessApp",
    "HarnessAppContext",
    "HarnessAppFactory",
    "HarnessAppSession",
    "HarnessSessionProvider",
    "HarnessSessionsAppProvider",
    "load_harness_app",
    "RuntimeAgent",
    "RuntimeSkills",
    "RuntimeTools",
]
