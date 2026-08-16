import importlib.util
import sys
from pathlib import Path
from typing import Any

from starlette.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
SERVER_PATH = ROOT / "scripts" / "docs_mcp_legacy_server.py"


def _load_server() -> Any:
    spec = importlib.util.spec_from_file_location("docs_mcp_legacy_server", SERVER_PATH)
    assert spec is not None
    loader = spec.loader
    assert loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["docs_mcp_legacy_server"] = module
    loader.exec_module(module)
    return module


server = _load_server()


def test_docs_legacy_server_supports_initialize_notifications_and_ping() -> None:
    client = TestClient(server.app)
    initialized = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": server.PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
    )

    assert initialized.status_code == 200
    assert initialized.headers["mcp-session-id"] == server.SESSION_ID
    assert initialized.json()["result"] == {
        "protocolVersion": server.PROTOCOL_VERSION,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": "Docs Legacy Remote", "version": "1.0.0"},
        "instructions": "Deterministic legacy MCP documentation fixture.",
    }

    notification = client.post(
        "/mcp",
        headers={"MCP-Session-Id": server.SESSION_ID},
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    ping = client.post(
        "/mcp",
        headers={"MCP-Session-Id": server.SESSION_ID},
        json={"jsonrpc": "2.0", "id": 2, "method": "ping"},
    )

    assert notification.status_code == 202
    assert ping.status_code == 200
    assert ping.json() == {"jsonrpc": "2.0", "id": 2, "result": {}}
