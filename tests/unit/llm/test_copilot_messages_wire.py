"""Copilot Messages authentication through the real SDK and a loopback gateway."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from email.parser import BytesHeaderParser
from urllib.parse import urlsplit

import anthropic
import pytest

from fast_agent.config import AnthropicSettings, Settings
from fast_agent.context import Context
from fast_agent.llm.provider.copilot.broker import CopilotEndpoint
from fast_agent.llm.provider.copilot.messages import CopilotMessagesLLM


def test_messages_wire_ignores_ambient_credentials() -> None:
    # A child process supplies hostile ambient state without patching SDKs or globals.
    env = dict(os.environ)
    for name in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_PROFILE",
        "ANTHROPIC_CONFIG_DIR",
    ):
        env[name] = "ambient-secret"
    env["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:1"
    env["ANTHROPIC_CUSTOM_HEADERS"] = "\n".join(
        f"{name}: ambient-secret"
        for name in (
            "Authorization",
            "aUtHoRiZaTiOn",
            "X-Api-Key",
            "Cookie",
            "Host",
            "Proxy-Authorization",
            "anthropic-version",
            "X-Ambient",
        )
    )
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        env[name] = env[name.lower()] = "http://127.0.0.1:1"
    env.pop("NO_PROXY", None)
    env.pop("no_proxy", None)
    result = subprocess.run(
        [sys.executable, __file__],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


async def _exercise_wire(status: int) -> None:
    requests: list[tuple[bytes, bytes]] = []

    async def respond(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        headers = BytesHeaderParser().parsebytes(head.split(b"\r\n", 1)[1])
        body = await reader.readexactly(int(headers["Content-Length"]))
        requests.append((head, body))
        # Redirect to the same simulator: following it would produce an extra request.
        writer.write(
            f"HTTP/1.1 {status} Simulated\r\n"
            "Content-Type: text/event-stream\r\n"
            "Content-Length: 0\r\n"
            "Connection: close\r\n"
            "Location: /stolen\r\n\r\n".encode()
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    environment = dict(os.environ)
    model = "claude-sonnet-5"
    llm = CopilotMessagesLLM(
        model=model,
        context=Context(
            config=Settings(
                anthropic=AnthropicSettings(
                    api_key="ambient-secret",
                    base_url="http://127.0.0.1:1",
                    default_headers={"Authorization": "ambient-secret"},
                )
            )
        ),
    )
    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        for generation, auth_header in enumerate(("authorization", "Authorization"), 1):
            llm._copilot_endpoint.set(
                CopilotEndpoint(
                    model_id=model,
                    wire_api="messages",
                    transport="sse",
                    base_url=f"http://127.0.0.1:{port}/v1",
                    headers={
                        auth_header: f"Bearer broker-{generation}",
                        "X-Broker": "trusted",
                    },
                )
            )
            requests.clear()
            async with llm._initialize_anthropic_client() as client:
                assert client.api_key == ""
                assert client.auth_token is None
                assert client.credentials is None

                async def stream_messages() -> None:
                    async with client.beta.messages.stream(
                        model=model,
                        max_tokens=16,
                        messages=[{"role": "user", "content": "hello"}],
                    ) as stream:
                        async for _ in stream:
                            pass

                if status == 200:
                    await stream_messages()
                else:
                    with pytest.raises(anthropic.APIStatusError) as error:
                        await stream_messages()
                    assert error.value.status_code == status

            assert len(requests) == 1
            head, body = requests[0]
            request_line, raw_headers = head.split(b"\r\n", 1)
            method, target, _ = request_line.split()
            assert method == b"POST"
            assert urlsplit(target).path == b"/v1/messages"
            headers = BytesHeaderParser().parsebytes(raw_headers)
            assert headers.get_all("Authorization") == [f"Bearer broker-{generation}"]
            assert headers["X-Api-Key"] is None
            assert headers["X-Broker"] == "trusted"
            assert headers["Host"] == f"127.0.0.1:{port}"
            assert headers["anthropic-version"]
            assert b"ambient-secret" not in head
            assert b"ambient-secret" not in body
            assert json.loads(body)["stream"] is True
            assert dict(os.environ) == environment


if __name__ == "__main__":
    # Ambient configuration is identical for each response; import the real SDK
    # once, while keeping a fresh loop, server, and provider for each scenario.
    for status in (200, 307, 308):
        asyncio.run(_exercise_wire(status))
