"""HTTP logging dependencies are loaded only when the transport starts."""

import subprocess
import sys


def test_http_logging_client_is_deferred_until_start() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import asyncio
import sys

from fast_agent.core.logging.transport import HTTPTransport

async def main():
    transport = HTTPTransport(endpoint="http://localhost:1/events")
    assert "aiohttp" not in sys.modules
    await transport.start()
    try:
        assert "aiohttp" in sys.modules
    finally:
        await transport.stop()

asyncio.run(main())
""",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
