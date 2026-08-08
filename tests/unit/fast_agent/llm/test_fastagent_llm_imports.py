from __future__ import annotations

import subprocess
import sys

import pytest


def test_provider_neutral_llm_import_defers_provider_sdks() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import fast_agent.llm.fastagent_llm; "
                "assert 'anthropic' not in sys.modules; "
                "assert 'openai' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("provider_module", "selected_sdk", "unused_sdk"),
    [
        ("fast_agent.llm.provider.openai.llm_openai", "openai", "anthropic"),
        ("fast_agent.llm.provider.anthropic.llm_anthropic", "anthropic", "openai"),
    ],
)
def test_provider_import_loads_only_selected_sdk(
    provider_module: str,
    selected_sdk: str,
    unused_sdk: str,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import {provider_module}; "
                "import sys; "
                f"assert {selected_sdk!r} in sys.modules; "
                f"assert {unused_sdk!r} not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
