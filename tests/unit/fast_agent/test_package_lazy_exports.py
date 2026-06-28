from __future__ import annotations

import importlib
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
