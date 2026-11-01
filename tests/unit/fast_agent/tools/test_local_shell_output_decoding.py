"""Local shell output that is not UTF-8 stays readable via the locale encoding."""

from __future__ import annotations

import shlex
import sys
from typing import TYPE_CHECKING

import pytest

from fast_agent.core.logging.logger import get_logger
from fast_agent.tools.execution_environment import ShellExecutionRequest
from fast_agent.tools.local_shell_executor import LocalEnvironment

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
@pytest.mark.parametrize("detach", [False, True], ids=["pipes", "spool"])
async def test_non_utf8_output_decodes_with_the_locale_encoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detach: bool
) -> None:
    # e.g. Python 3.12-3.14 on a cp1252 Windows locale writes its pipes in cp1252.
    monkeypatch.setattr("fast_agent.utils.text.locale.getencoding", lambda: "cp1252")
    script = "import sys; sys.stdout.buffer.write(b'caf\\xe9\\n' + 'café\\n'.encode())"
    environment = LocalEnvironment(logger=get_logger(__name__), working_directory=tmp_path)

    execution = await environment.execute(
        ShellExecutionRequest(
            command=f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}",
            detach=detach,
            terminate_on_cancel=not detach,
        )
    )

    assert execution.result.stdout.splitlines() == ["café", "café"]
