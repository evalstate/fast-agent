import asyncio
import os
import sys
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path

import pytest
import pytest_asyncio

from fast_agent import FastAgent
from fast_agent.core.logging.logger import LoggingConfig
from fast_agent.core.logging.transport import AsyncEventBus


@pytest_asyncio.fixture(autouse=True)
async def cleanup_event_bus() -> AsyncIterator[None]:
    """Gracefully stop loop-owned logging tasks between integration tests."""
    yield
    await LoggingConfig.shutdown()
    AsyncEventBus.reset()


@pytest.fixture(autouse=True)
def isolate_fast_agent_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep integration tests independent of the developer's fast-agent home."""
    from fast_agent import config as config_module

    config_module._settings = None
    for key in ("FAST_AGENT_HOME", "FAST_AGENT_RUNTIME_HOME"):
        monkeypatch.delenv(key, raising=False)
    yield
    config_module._settings = None


@pytest.fixture(scope="session")
def mcp_test_ports(unused_tcp_port_factory: Callable[[], int]) -> Iterator[dict[str, int]]:
    # Independent workers and simultaneous suite runs must not share fixed ports.
    ports = {name: unused_tcp_port_factory() for name in ("sse", "http", "request_http")}

    env_updates = {
        "FAST_AGENT_TEST_SSE_PORT": str(ports["sse"]),
        "FAST_AGENT_TEST_HTTP_PORT": str(ports["http"]),
    }
    previous = {key: os.getenv(key) for key in env_updates}
    os.environ.update(env_updates)

    yield ports

    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def wait_for_port():
    async def _wait_for_port(
        host: str,
        port: int,
        *,
        process=None,
        timeout: float = 5.0,
        interval: float = 0.1,
    ) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            if process is not None and process.poll() is not None:
                stdout = ""
                stderr = ""
                if process.stdout or process.stderr:
                    stdout, stderr = process.communicate(timeout=1)
                raise AssertionError(f"Server exited early. stdout={stdout!r} stderr={stderr!r}")
            try:
                reader, writer = await asyncio.open_connection(host, port)
                writer.close()
                await writer.wait_closed()
                return
            except OSError as error:
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError(
                        f"Server did not start listening on {host}:{port}"
                    ) from error
                await asyncio.sleep(interval)

    return _wait_for_port


# Set the project root directory for tests
@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory as a Path object"""
    # Go up from tests/e2e directory to find project root
    return Path(__file__).parent.parent.parent


@pytest.fixture
def fast_agent_factory(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    mcp_test_ports,
) -> Callable[[str], FastAgent]:
    """Load test-local configs with exception-safe cwd and argv restoration."""
    test_dir = request.path.parent
    monkeypatch.chdir(test_dir)

    def create(config_name: str) -> FastAgent:
        with monkeypatch.context() as arguments:
            arguments.setattr(sys, "argv", [sys.argv[0]])
            return FastAgent(
                "Test Agent",
                config_path=str(test_dir / config_name),
                ignore_unknown_args=True,
            )

    return create


@pytest.fixture
def fast_agent(fast_agent_factory: Callable[[str], FastAgent]) -> FastAgent:
    return fast_agent_factory("fastagent.config.yaml")


@pytest.fixture
def markup_fast_agent(fast_agent_factory: Callable[[str], FastAgent]) -> FastAgent:
    return fast_agent_factory("fastagent.config.markup.yaml")


@pytest.fixture
def auto_sampling_off_fast_agent(fast_agent_factory: Callable[[str], FastAgent]) -> FastAgent:
    return fast_agent_factory("fastagent.config.auto_sampling_off.yaml")
