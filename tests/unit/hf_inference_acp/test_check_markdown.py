from __future__ import annotations

import sys
from pathlib import Path


def _ensure_hf_inference_acp_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    package_root = repo_root / "publish" / "hf-inference-acp" / "src"
    sys.path.insert(0, str(package_root))


def test_render_setup_check_markdown_is_consistent_with_command_markdown() -> None:
    _ensure_hf_inference_acp_on_path()

    from hf_inference_acp.agents import (  # ty: ignore[unresolved-import]
        _render_setup_check_markdown,
    )

    rendered = _render_setup_check_markdown(
        hf_inference_acp_version="1.2.3",
        fast_agent_version="4.5.6",
        huggingface_hub_version="7.8.9",
        hf_token_source="env",
        config_exists=True,
        default_model="hf.openai/gpt-oss-20b",
    )

    assert rendered.startswith("# check\n")
    assert "## Runtime" in rendered
    assert "- **hf-inference-acp**: `1.2.3`" in rendered
    assert "- **fast-agent-mcp**: `4.5.6`" in rendered
    assert "- **huggingface_hub**: `7.8.9`" in rendered
    assert "## Authentication" in rendered
    assert "- **HF_TOKEN**: set (`env`)" in rendered
    assert "## Configuration" in rendered
    assert "  - **Default model**: `hf.openai/gpt-oss-20b`" in rendered


def test_render_setup_check_markdown_handles_missing_dependencies_and_token() -> None:
    _ensure_hf_inference_acp_on_path()

    from hf_inference_acp.agents import (  # ty: ignore[unresolved-import]
        _render_setup_check_markdown,
    )

    rendered = _render_setup_check_markdown(
        hf_inference_acp_version=None,
        fast_agent_version=None,
        huggingface_hub_version=None,
        hf_token_source=None,
        config_exists=False,
        default_model="hf.moonshotai/Kimi-K2-Instruct-0905:groq",
    )

    assert "- **hf-inference-acp**: `unknown`" in rendered
    assert "- **fast-agent-mcp**: `unknown`" in rendered
    assert "- **huggingface_hub**: not installed" in rendered
    assert "Install with `uv tool install -U huggingface_hub`" in rendered
    assert "- **HF_TOKEN**: not set" in rendered
    assert "  - **Status**: will be created on first use" in rendered
