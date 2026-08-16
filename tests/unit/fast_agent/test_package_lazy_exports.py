from __future__ import annotations

import importlib
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def restore_fast_agent_import_state() -> Iterator[None]:
    original_modules = {
        name: module for name, module in sys.modules.items() if name.startswith("fast_agent")
    }
    yield
    for name in list(sys.modules):
        if name.startswith("fast_agent"):
            sys.modules.pop(name, None)
    sys.modules.update(original_modules)


def test_package_import_defers_config_until_public_export_access() -> None:
    sys.modules.pop("fast_agent", None)
    sys.modules.pop("fast_agent.config", None)

    fast_agent = importlib.import_module("fast_agent")

    assert "fast_agent.config" not in sys.modules
    assert fast_agent.Settings.__name__ == "Settings"
    assert "fast_agent.config" in sys.modules


def test_package_import_defers_types_until_public_export_access() -> None:
    sys.modules.pop("fast_agent", None)
    sys.modules.pop("fast_agent.types", None)

    fast_agent = importlib.import_module("fast_agent")

    assert "fast_agent.types" not in sys.modules
    assert fast_agent.RequestParams.__name__ == "RequestParams"
    assert "fast_agent.types" in sys.modules


def test_a2a_package_import_defers_server_stack() -> None:
    sys.modules.pop("fast_agent.a2a", None)
    sys.modules.pop("fast_agent.a2a.server", None)

    importlib.import_module("fast_agent.a2a")

    assert "fast_agent.a2a.server" not in sys.modules


def test_a2a_connect_import_defers_config() -> None:
    sys.modules.pop("fast_agent.a2a.connect", None)
    sys.modules.pop("fast_agent.config", None)

    connect = importlib.import_module("fast_agent.a2a.connect")

    assert "fast_agent.config" not in sys.modules
    assert connect.normalize_a2a_transport("jsonrpc") == "JSONRPC"


def test_commands_option_parsing_import_defers_command_runtime() -> None:
    sys.modules.pop("fast_agent.commands", None)
    sys.modules.pop("fast_agent.commands.context", None)

    importlib.import_module("fast_agent.commands.option_parsing")

    assert "fast_agent.commands.context" not in sys.modules

    commands = importlib.import_module("fast_agent.commands")
    assert set(commands.__all__) <= set(dir(commands))
    assert commands.CommandContext.__name__ == "CommandContext"
    assert commands.__dict__["CommandContext"] is commands.CommandContext
    assert "fast_agent.commands.context" in sys.modules
    assert all(getattr(commands, name) is not None for name in commands.__all__)
    with pytest.raises(AttributeError):
        commands.missing_export


def test_mcp_connect_targets_import_defers_content_helpers() -> None:
    sys.modules.pop("fast_agent.mcp", None)
    sys.modules.pop("fast_agent.mcp.helpers", None)
    sys.modules.pop("fast_agent.mcp.helpers.content_helpers", None)

    connect = importlib.import_module("fast_agent.mcp.connect_targets")

    assert "fast_agent.mcp.helpers" not in sys.modules
    assert "fast_agent.mcp.helpers.content_helpers" not in sys.modules
    assert connect.resolve_connect_auth_token("token") == "token"

    mcp = importlib.import_module("fast_agent.mcp")
    assert set(mcp.__all__) <= set(dir(mcp))
    assert mcp.get_text.__name__ == "get_text"
    assert mcp.__dict__["get_text"] is mcp.get_text
    assert "fast_agent.mcp.helpers" in sys.modules
    assert all(getattr(mcp, name) is not None for name in mcp.__all__)
    with pytest.raises(AttributeError):
        mcp.missing_export


def test_mcp_auth_context_import_defers_server_auth_stack() -> None:
    sys.modules.pop("fast_agent.mcp.auth", None)
    sys.modules.pop("fast_agent.mcp.auth.context", None)
    sys.modules.pop("fast_agent.mcp.auth.huggingface", None)
    sys.modules.pop("fast_agent.mcp.auth.middleware", None)

    context = importlib.import_module("fast_agent.mcp.auth.context")

    assert "fast_agent.mcp.auth.huggingface" not in sys.modules
    assert "fast_agent.mcp.auth.middleware" not in sys.modules

    auth = importlib.import_module("fast_agent.mcp.auth")
    assert set(auth.__all__) <= set(dir(auth))
    assert auth.request_bearer_token is context.request_bearer_token
    assert "fast_agent.mcp.auth.huggingface" not in sys.modules
    assert "fast_agent.mcp.auth.middleware" not in sys.modules

    assert auth.HuggingFaceOAuthOrHubTokenVerifier.__name__ == (
        "HuggingFaceOAuthOrHubTokenVerifier"
    )
    assert auth.HFAuthHeaderMiddleware.__name__ == "HFAuthHeaderMiddleware"
    assert "fast_agent.mcp.auth.huggingface" in sys.modules
    assert "fast_agent.mcp.auth.middleware" in sys.modules


def test_hooks_type_import_defers_hook_runtime() -> None:
    sys.modules.pop("fast_agent.hooks", None)
    sys.modules.pop("fast_agent.hooks.lifecycle_hook_types", None)
    sys.modules.pop("fast_agent.hooks.compaction", None)

    importlib.import_module("fast_agent.hooks.lifecycle_hook_types")

    assert "fast_agent.hooks.compaction" not in sys.modules

    hooks = importlib.import_module("fast_agent.hooks")
    assert set(hooks.__all__) <= set(dir(hooks))
    assert hooks.auto_compact_history.__name__ == "auto_compact_history"
    assert hooks.__dict__["auto_compact_history"] is hooks.auto_compact_history
    assert "fast_agent.hooks.compaction" in sys.modules
    assert all(getattr(hooks, name) is not None for name in hooks.__all__)
    with pytest.raises(AttributeError):
        hooks.missing_export


def test_runtime_request_import_defers_llm_request_params() -> None:
    sys.modules.pop("fast_agent.cli.runtime", None)
    sys.modules.pop("fast_agent.cli.runtime.run_request", None)
    sys.modules.pop("fast_agent.llm.request_params", None)

    runtime = importlib.import_module("fast_agent.cli.runtime")

    assert runtime.AgentRunRequest.__name__ == "AgentRunRequest"
    assert "fast_agent.llm.request_params" not in sys.modules


def test_agent_config_import_defers_mcp_client_session() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "from fast_agent.agents.agent_types import AgentConfig; "
                "assert AgentConfig(name='test').name == 'test'; "
                "assert 'mcp.client.session' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
