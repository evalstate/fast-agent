from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

import pytest_asyncio
from acp.schema import ClientCapabilities, FileSystemCapability, Implementation, InitializeResponse
from acp.stdio import spawn_agent_process

TEST_DIR = Path(__file__).parent
if str(TEST_DIR) not in sys.path:
    sys.path.append(str(TEST_DIR))

from test_client import TestClient  # noqa: E402

if TYPE_CHECKING:
    from acp.client.connection import ClientSideConnection

CONFIG_PATH = TEST_DIR / "fastagent.config.yaml"


def _fast_agent_cmd(
    name: str,
    *,
    servers: tuple[str, ...] = (),
    no_permissions: bool = False,
    shell: bool = False,
) -> tuple[str, ...]:
    cmd = [
        sys.executable,
        "-m",
        "fast_agent.cli",
        "serve",
        "--config-path",
        str(CONFIG_PATH),
        "--transport",
        "acp",
        "--model",
        "passthrough",
        "--name",
        name,
    ]
    if servers:
        cmd.extend(["--servers", *servers])
    if no_permissions:
        cmd.append("--no-permissions")
    if shell:
        cmd.append("--shell")
    return tuple(cmd)


@asynccontextmanager
async def _spawn_initialized_agent(
    cmd: tuple[str, ...],
    *,
    terminal: bool,
    fs_read: bool = True,
    fs_write: bool = True,
    client_name: str = "pytest-client",
    client_version: str = "0.0.1",
) -> AsyncIterator[tuple[ClientSideConnection, TestClient, InitializeResponse]]:
    client = TestClient()
    async with spawn_agent_process(lambda _: client, *cmd) as (connection, _process):
        init_response = await connection.initialize(
            protocol_version=1,
            client_capabilities=ClientCapabilities(
                fs=FileSystemCapability(read_text_file=fs_read, write_text_file=fs_write),
                terminal=terminal,
            ),
            client_info=Implementation(name=client_name, version=client_version),
        )
        yield connection, client, init_response


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def acp_basic_process() -> AsyncIterator[tuple[ClientSideConnection, TestClient, InitializeResponse]]:
    cmd = _fast_agent_cmd("fast-agent-acp-test")
    async with _spawn_initialized_agent(cmd, terminal=False) as harness:
        yield harness


@pytest_asyncio.fixture
async def acp_basic(
    acp_basic_process: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> AsyncIterator[tuple[ClientSideConnection, TestClient, InitializeResponse]]:
    connection, client, init_response = acp_basic_process
    client.reset()
    yield connection, client, init_response


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def acp_content_process() -> AsyncIterator[
    tuple[ClientSideConnection, TestClient, InitializeResponse]
]:
    cmd = _fast_agent_cmd("fast-agent-acp-content-test")
    async with _spawn_initialized_agent(cmd, terminal=False) as harness:
        yield harness


@pytest_asyncio.fixture
async def acp_content(
    acp_content_process: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> AsyncIterator[tuple[ClientSideConnection, TestClient, InitializeResponse]]:
    connection, client, init_response = acp_content_process
    client.reset()
    yield connection, client, init_response


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def acp_filesystem_toolcall_process() -> AsyncIterator[
    tuple[ClientSideConnection, TestClient, InitializeResponse]
]:
    cmd = _fast_agent_cmd(
        "fast-agent-acp-filesystem-toolcall-test",
        no_permissions=True,
    )
    async with _spawn_initialized_agent(
        cmd,
        terminal=False,
        client_name="pytest-filesystem-client",
    ) as harness:
        yield harness


@pytest_asyncio.fixture
async def acp_filesystem_toolcall(
    acp_filesystem_toolcall_process: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> AsyncIterator[tuple[ClientSideConnection, TestClient, InitializeResponse]]:
    connection, client, init_response = acp_filesystem_toolcall_process
    client.reset()
    yield connection, client, init_response


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def acp_permissions_process() -> AsyncIterator[
    tuple[ClientSideConnection, TestClient, InitializeResponse]
]:
    cmd = _fast_agent_cmd(
        "fast-agent-acp-test",
        servers=("progress_test",),
    )
    async with _spawn_initialized_agent(cmd, terminal=False) as harness:
        yield harness


@pytest_asyncio.fixture
async def acp_permissions(
    acp_permissions_process: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> AsyncIterator[tuple[ClientSideConnection, TestClient, InitializeResponse]]:
    connection, client, init_response = acp_permissions_process
    client.reset()
    yield connection, client, init_response


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def acp_permissions_no_perms_process() -> AsyncIterator[
    tuple[ClientSideConnection, TestClient, InitializeResponse]
]:
    cmd = _fast_agent_cmd(
        "fast-agent-acp-test",
        servers=("progress_test",),
        no_permissions=True,
    )
    async with _spawn_initialized_agent(cmd, terminal=False) as harness:
        yield harness


@pytest_asyncio.fixture
async def acp_permissions_no_perms(
    acp_permissions_no_perms_process: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> AsyncIterator[tuple[ClientSideConnection, TestClient, InitializeResponse]]:
    connection, client, init_response = acp_permissions_no_perms_process
    client.reset()
    yield connection, client, init_response


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def acp_runtime_telemetry_shell_process() -> AsyncIterator[
    tuple[ClientSideConnection, TestClient, InitializeResponse]
]:
    cmd = _fast_agent_cmd(
        "fast-agent-acp-runtime-telemetry-test",
        no_permissions=True,
        shell=True,
    )
    async with _spawn_initialized_agent(
        cmd,
        terminal=True,
        client_name="pytest-telemetry-client",
    ) as harness:
        yield harness


@pytest_asyncio.fixture
async def acp_runtime_telemetry_shell(
    acp_runtime_telemetry_shell_process: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> AsyncIterator[tuple[ClientSideConnection, TestClient, InitializeResponse]]:
    connection, client, init_response = acp_runtime_telemetry_shell_process
    client.reset()
    yield connection, client, init_response


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def acp_runtime_telemetry_process() -> AsyncIterator[
    tuple[ClientSideConnection, TestClient, InitializeResponse]
]:
    cmd = _fast_agent_cmd(
        "fast-agent-acp-runtime-telemetry-test",
        no_permissions=True,
    )
    async with _spawn_initialized_agent(
        cmd,
        terminal=False,
        client_name="pytest-telemetry-client",
    ) as harness:
        yield harness


@pytest_asyncio.fixture
async def acp_runtime_telemetry(
    acp_runtime_telemetry_process: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> AsyncIterator[tuple[ClientSideConnection, TestClient, InitializeResponse]]:
    connection, client, init_response = acp_runtime_telemetry_process
    client.reset()
    yield connection, client, init_response


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def acp_tool_notifications_process() -> AsyncIterator[
    tuple[ClientSideConnection, TestClient, InitializeResponse]
]:
    cmd = _fast_agent_cmd(
        "fast-agent-acp-test",
        servers=("progress_test",),
        no_permissions=True,
    )
    async with _spawn_initialized_agent(cmd, terminal=False) as harness:
        yield harness


@pytest_asyncio.fixture
async def acp_tool_notifications(
    acp_tool_notifications_process: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> AsyncIterator[tuple[ClientSideConnection, TestClient, InitializeResponse]]:
    connection, client, init_response = acp_tool_notifications_process
    client.reset()
    yield connection, client, init_response


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def acp_terminal_shell_process() -> AsyncIterator[
    tuple[ClientSideConnection, TestClient, InitializeResponse]
]:
    cmd = _fast_agent_cmd(
        "fast-agent-acp-terminal-test",
        shell=True,
    )
    async with _spawn_initialized_agent(
        cmd,
        terminal=True,
        client_name="pytest-terminal-client",
    ) as harness:
        yield harness


@pytest_asyncio.fixture
async def acp_terminal_shell(
    acp_terminal_shell_process: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> AsyncIterator[tuple[ClientSideConnection, TestClient, InitializeResponse]]:
    connection, client, init_response = acp_terminal_shell_process
    client.reset()
    yield connection, client, init_response


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def acp_terminal_no_shell_process() -> AsyncIterator[
    tuple[ClientSideConnection, TestClient, InitializeResponse]
]:
    cmd = _fast_agent_cmd("fast-agent-acp-terminal-test")
    async with _spawn_initialized_agent(
        cmd,
        terminal=True,
        client_name="pytest-terminal-client",
    ) as harness:
        yield harness


@pytest_asyncio.fixture
async def acp_terminal_no_shell(
    acp_terminal_no_shell_process: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> AsyncIterator[tuple[ClientSideConnection, TestClient, InitializeResponse]]:
    connection, client, init_response = acp_terminal_no_shell_process
    client.reset()
    yield connection, client, init_response


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def acp_terminal_client_unsupported_process() -> AsyncIterator[
    tuple[ClientSideConnection, TestClient, InitializeResponse]
]:
    cmd = _fast_agent_cmd(
        "fast-agent-acp-terminal-test",
        shell=True,
    )
    async with _spawn_initialized_agent(
        cmd,
        terminal=False,
        client_name="pytest-terminal-client",
    ) as harness:
        yield harness


@pytest_asyncio.fixture
async def acp_terminal_client_unsupported(
    acp_terminal_client_unsupported_process: tuple[ClientSideConnection, TestClient, InitializeResponse],
) -> AsyncIterator[tuple[ClientSideConnection, TestClient, InitializeResponse]]:
    connection, client, init_response = acp_terminal_client_unsupported_process
    client.reset()
    yield connection, client, init_response
