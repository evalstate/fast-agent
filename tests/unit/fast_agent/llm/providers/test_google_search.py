"""Tests for Google grounding with Google Search support."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from fast_agent.config import Settings
from fast_agent.context import Context
from fast_agent.llm.provider.google.llm_google_native import GoogleNativeLLM


def _build_llm(config: Settings | None = None, **kwargs) -> GoogleNativeLLM:
    config = config or Settings()
    return GoogleNativeLLM(context=Context(config=config), **kwargs)


@pytest.mark.unit
def test_google_search_supported_models() -> None:
    """Grounding with Google Search should be supported on Gemini 2.5 and Gemini 3/3.5 models."""
    for model in (
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-3-pro-preview",
        "gemini-3.5-flash",
        "gemini-2.0-flash",
    ):
        llm = _build_llm(model=model)
        assert llm.web_search_supported is True, f"Google Search should be supported on {model}"


@pytest.mark.unit
def test_google_search_toggle_state() -> None:
    """Toggling Google Search state should store state and fail if unsupported."""
    llm = _build_llm(model="gemini-3.5-flash")
    assert llm.web_search_enabled is False

    llm.set_web_search_enabled(True)
    assert llm.web_search_enabled is True

    llm.set_web_search_enabled(False)
    assert llm.web_search_enabled is False

    llm.set_web_search_enabled(None)
    assert llm.web_search_enabled is False


@pytest.mark.unit
def test_apply_citations_formatting() -> None:
    """Check that _apply_citations correctly formats grounding metadata segments."""
    llm = _build_llm(model="gemini-3.5-flash")

    # Construct mock grounding metadata
    mock_chunk_1 = MagicMock()
    mock_chunk_1.web = MagicMock(uri="https://aljazeera.com")
    
    mock_chunk_2 = MagicMock()
    mock_chunk_2.web = MagicMock(uri="https://uefa.com")

    mock_support_1 = MagicMock()
    mock_support_1.segment = MagicMock(end_index=23)
    mock_support_1.grounding_chunk_indices = [0]

    mock_support_2 = MagicMock()
    mock_support_2.segment = MagicMock(end_index=47)
    mock_support_2.grounding_chunk_indices = [0, 1]

    mock_metadata = MagicMock()
    mock_metadata.grounding_supports = [mock_support_1, mock_support_2]
    mock_metadata.grounding_chunks = [mock_chunk_1, mock_chunk_2]

    raw_text_2 = "Sentence one ends here. Sentence two ends here."

    formatted = llm._apply_citations(raw_text_2, mock_metadata)
    assert "[1](https://aljazeera.com)" in formatted
    assert "[2](https://uefa.com)" in formatted


@pytest.mark.unit
@pytest.mark.asyncio
async def test_google_completion_injects_search_tool() -> None:
    """Verify that _google_completion adds the GoogleSearch tool when web_search is enabled."""
    llm = _build_llm(model="gemini-3.5-flash")
    llm.set_web_search_enabled(True)

    # Mock the client generate_content call
    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.content = MagicMock(parts=[MagicMock(text="Hello", thought=False, function_call=None)])
    mock_candidate.finish_reason = "STOP"
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = None

    mock_client = MagicMock()
    mock_client.aio = AsyncMock()
    mock_client.aio.models = AsyncMock()
    mock_client.aio.models.generate_content.return_value = mock_response

    # Mock _initialize_google_client, we don't need real API keys
    with patch.object(llm, "_initialize_google_client", return_value=mock_client), \
         patch.object(llm, "_stream_generate_content", return_value=mock_response) as mock_stream_gen:
        
        await llm._google_completion(message=[])
        
        # Capture the GenerateContentConfig object passed to the API
        called_args, called_kwargs = mock_stream_gen.call_args
        config_passed = called_kwargs.get("config")
        
        assert config_passed is not None
        assert len(config_passed.tools) == 1
        assert getattr(config_passed.tools[0], "google_search", None) is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_google_completion_injects_search_tool_with_custom_tools() -> None:
    """Verify include_server_side_tool_invocations on ToolConfig when custom tools are active."""
    llm = _build_llm(model="gemini-3.5-flash")
    llm.set_web_search_enabled(True)

    mock_response = MagicMock()
    mock_candidate = MagicMock()
    mock_candidate.content = MagicMock(parts=[MagicMock(text="Hello", thought=False, function_call=None)])
    mock_candidate.finish_reason = "STOP"
    mock_response.candidates = [mock_candidate]
    mock_response.usage_metadata = None

    mock_client = MagicMock()
    mock_client.aio = AsyncMock()
    mock_client.aio.models = AsyncMock()
    mock_client.aio.models.generate_content.return_value = mock_response

    custom_tool = MagicMock()
    custom_tool.name = "my_tool"
    custom_tool.description = "custom tool"
    custom_tool.inputSchema = {}

    with patch.object(llm, "_initialize_google_client", return_value=mock_client), \
         patch.object(llm, "_stream_generate_content", return_value=mock_response) as mock_stream_gen:
        
        await llm._google_completion(message=[], tools=[custom_tool])
        
        called_args, called_kwargs = mock_stream_gen.call_args
        config_passed = called_kwargs.get("config")
        
        assert config_passed is not None
        assert config_passed.tool_config is not None
        assert config_passed.tool_config.include_server_side_tool_invocations is True
