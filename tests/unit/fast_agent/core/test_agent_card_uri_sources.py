from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

import pytest

from fast_agent import FastAgent
from fast_agent.agents.agent_types import AgentConfig
from fast_agent.core.exceptions import AgentConfigError

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _card_text(name: str, *, model: str | None = None) -> str:
    model_lines = [f"model: {model}"] if model else []
    return "\n".join(
        [
            "---",
            "type: agent",
            f"name: {name}",
            *model_lines,
            "---",
            "Return ok.",
            "",
        ]
    )


@dataclass
class _CardServer:
    server: ThreadingHTTPServer
    thread: threading.Thread
    responses: dict[str, str] = field(default_factory=dict)

    @property
    def base_url(self) -> str:
        host = str(self.server.server_address[0])
        port = int(self.server.server_address[1])
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def card_server() -> Iterator[_CardServer]:
    responses: dict[str, str] = {}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = responses.get(self.path)
            if body is None:
                self.send_response(404)
                self.end_headers()
                return
            encoded = body.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            del format, args

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    card_server = _CardServer(server=server, thread=thread, responses=responses)
    try:
        yield card_server
    finally:
        card_server.close()


def test_load_agents_supports_file_uri_agent_card(tmp_path: Path) -> None:
    card_path = tmp_path / "file_agent.md"
    card_path.write_text(_card_text("file_agent"), encoding="utf-8")
    fast = FastAgent("card-uri-test", parse_cli_args=False, quiet=True)

    loaded_names = fast.load_agents(card_path.as_uri())

    assert loaded_names == ["file_agent"]
    assert "file_agent" in fast.agents


def test_load_agents_supports_remote_agent_card(card_server: _CardServer) -> None:
    card_server.responses["/remote_agent.md"] = _card_text("remote_agent")
    fast = FastAgent("card-uri-test", parse_cli_args=False, quiet=True)

    loaded_names = fast.load_agents(f"{card_server.base_url}/remote_agent.md")

    assert loaded_names == ["remote_agent"]
    assert "remote_agent" in fast.agents


def test_load_agents_defaults_extensionless_remote_agent_card_to_markdown(
    card_server: _CardServer,
) -> None:
    card_server.responses["/remote_agent"] = _card_text("remote_agent", model="passthrough")
    fast = FastAgent("card-uri-test", parse_cli_args=False, quiet=True)

    loaded_names = fast.load_agents(f"{card_server.base_url}/remote_agent")

    assert loaded_names == ["remote_agent"]
    config = fast.agents["remote_agent"]["config"]
    assert isinstance(config, AgentConfig)
    assert config.model == "passthrough"


def test_remote_yaml_cards_preserve_environment_placeholders(card_server: _CardServer) -> None:
    card_server.responses["/remote.yaml"] = "\n".join(
        [
            'name: "${REMOTE_AGENT_NAME}"',
            "mcp_connect:",
            '  - target: "@example/server"',
            "    headers:",
            '      Authorization: "Bearer ${REMOTE_CARD_TOKEN}"',
            "",
        ]
    )
    original_name = os.environ.get("REMOTE_AGENT_NAME")
    original_token = os.environ.get("REMOTE_CARD_TOKEN")
    os.environ["REMOTE_AGENT_NAME"] = "interpolated_name"
    os.environ["REMOTE_CARD_TOKEN"] = "interpolated_token"
    try:
        fast = FastAgent("card-uri-test", parse_cli_args=False, quiet=True)
        loaded_names = fast.load_agents(f"{card_server.base_url}/remote.yaml")

        assert loaded_names == ["${REMOTE_AGENT_NAME}"]
        config = fast.agents["${REMOTE_AGENT_NAME}"]["config"]
        assert config.mcp_connect[0].headers == {"Authorization": "Bearer ${REMOTE_CARD_TOKEN}"}
    finally:
        if original_name is None:
            os.environ.pop("REMOTE_AGENT_NAME")
        else:
            os.environ["REMOTE_AGENT_NAME"] = original_name
        if original_token is None:
            os.environ.pop("REMOTE_CARD_TOKEN")
        else:
            os.environ["REMOTE_CARD_TOKEN"] = original_token


def test_remote_frontmatter_preserves_environment_and_escaped_file_directive(
    card_server: _CardServer,
) -> None:
    card_server.responses["/remote.md"] = "\n".join(
        [
            "---",
            'name: "${REMOTE_AGENT_NAME}"',
            "---",
            r"\{{file:instructions.md}}",
            "",
        ]
    )
    original_name = os.environ.get("REMOTE_AGENT_NAME")
    os.environ["REMOTE_AGENT_NAME"] = "interpolated_name"
    try:
        fast = FastAgent("card-uri-test", parse_cli_args=False, quiet=True)
        fast.load_agents(f"{card_server.base_url}/remote.md")

        config = fast.agents["${REMOTE_AGENT_NAME}"]["config"]
        assert config.instruction == r"\{{file:instructions.md}}"
    finally:
        if original_name is None:
            os.environ.pop("REMOTE_AGENT_NAME")
        else:
            os.environ["REMOTE_AGENT_NAME"] = original_name


@pytest.mark.parametrize("directive", ["{{file:secret.md}}", "{{file_silent:secret.md}}"])
def test_remote_cards_reject_file_instruction_directives(
    card_server: _CardServer,
    directive: str,
) -> None:
    card_server.responses["/remote.md"] = "\n".join(
        [
            "---",
            "name: remote_agent",
            "---",
            directive,
            "",
        ]
    )
    source = f"{card_server.base_url}/remote.md"
    fast = FastAgent("card-uri-test", parse_cli_args=False, quiet=True)

    with pytest.raises(AgentConfigError, match="file instruction directives") as exc_info:
        fast.load_agents(source)

    assert source in str(exc_info.value)


def test_remote_url_template_cannot_smuggle_file_directive(card_server: _CardServer) -> None:
    card_server.responses["/included.txt"] = "{{file:secret.md}}"
    card_server.responses["/remote.yaml"] = "\n".join(
        [
            "name: remote_agent",
            f'instruction: "{{{{url:{card_server.base_url}/included.txt}}}}"',
            "",
        ]
    )
    source = f"{card_server.base_url}/remote.yaml"
    fast = FastAgent("card-uri-test", parse_cli_args=False, quiet=True)

    with pytest.raises(AgentConfigError, match="file instruction directives") as exc_info:
        fast.load_agents(source)

    assert source in str(exc_info.value)


def test_remote_nested_url_template_cannot_smuggle_file_directive(
    card_server: _CardServer,
) -> None:
    card_server.responses["/first.txt"] = f"{{{{url:{card_server.base_url}/second.txt}}}}"
    card_server.responses["/second.txt"] = "{{file:secret.md}}"
    card_server.responses["/remote.yaml"] = "\n".join(
        [
            "name: remote_agent",
            f'instruction: "{{{{url:{card_server.base_url}/first.txt}}}}"',
            "",
        ]
    )
    source = f"{card_server.base_url}/remote.yaml"
    fast = FastAgent("card-uri-test", parse_cli_args=False, quiet=True)

    with pytest.raises(AgentConfigError, match="file instruction directives") as exc_info:
        fast.load_agents(source)

    assert source in str(exc_info.value)


def test_remote_url_template_rejects_excessive_nesting(card_server: _CardServer) -> None:
    for index in range(11):
        card_server.responses[f"/include-{index}.txt"] = (
            f"{{{{url:{card_server.base_url}/include-{index + 1}.txt}}}}"
        )
    card_server.responses["/include-11.txt"] = "Never reached."
    card_server.responses["/remote.yaml"] = "\n".join(
        [
            "name: remote_agent",
            f'instruction: "{{{{url:{card_server.base_url}/include-0.txt}}}}"',
            "",
        ]
    )
    source = f"{card_server.base_url}/remote.yaml"
    fast = FastAgent("card-uri-test", parse_cli_args=False, quiet=True)

    with pytest.raises(AgentConfigError, match="URL include depth exceeded") as exc_info:
        fast.load_agents(source)

    assert source in str(exc_info.value)


def test_remote_escaped_url_template_remains_literal(card_server: _CardServer) -> None:
    source = f"{card_server.base_url}/remote.yaml"
    card_server.responses["/remote.yaml"] = "\n".join(
        [
            "name: remote_agent",
            f"instruction: '\\{{{{url:{card_server.base_url}/not-fetched.txt}}}}'",
            "",
        ]
    )
    fast = FastAgent("card-uri-test", parse_cli_args=False, quiet=True)

    fast.load_agents(source)

    config = fast.agents["remote_agent"]["config"]
    assert config.instruction == f"\\{{{{url:{card_server.base_url}/not-fetched.txt}}}}"


@pytest.mark.parametrize("messages", ["/history.json", "history.json"])
def test_remote_cards_reject_messages_field(
    card_server: _CardServer,
    messages: str,
) -> None:
    card_server.responses["/remote.yaml"] = f"name: remote_agent\nmessages: {messages}\n"
    source = f"{card_server.base_url}/remote.yaml"
    fast = FastAgent("card-uri-test", parse_cli_args=False, quiet=True)

    with pytest.raises(AgentConfigError, match="cannot use the 'messages' field") as exc_info:
        fast.load_agents(source)

    assert source in str(exc_info.value)
