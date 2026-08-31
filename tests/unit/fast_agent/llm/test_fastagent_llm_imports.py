from __future__ import annotations

import subprocess
import sys

import pytest


def test_provider_neutral_runtime_import_defers_optional_dependencies() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import fast_agent.llm.fastagent_llm; "
                "import fast_agent.agents.mcp_agent; "
                "assert 'anthropic' not in sys.modules; "
                "assert 'openai' not in sys.modules; "
                "assert 'fast_agent.agents.shell_runtime' not in sys.modules; "
                "assert 'fast_agent.agents.environment_filesystem_runtime' "
                "not in sys.modules; "
                "assert 'fast_agent.agents.local_filesystem_runtime' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_provider_neutral_llm_import_defers_huggingface_discovery() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import fast_agent.llm.fastagent_llm; "
                "assert 'fast_agent.llm.hf_inference_lookup' not in sys.modules; "
                "assert 'huggingface_hub' not in sys.modules; "
                "from fast_agent.llm import lookup_inference_providers; "
                "assert callable(lookup_inference_providers); "
                "assert 'fast_agent.llm.hf_inference_lookup' in sys.modules; "
                "assert 'huggingface_hub' in sys.modules"
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
