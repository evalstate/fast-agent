"""Fast requires OAuth independently of the ordinary xAI API-key path."""

from unittest.mock import Mock

import pytest

from fast_agent.auth.credentials import OAuthCredential
from fast_agent.cli.runtime.model_bootstrap import activate_model_picker_provider
from fast_agent.commands.context import AgentProvider
from fast_agent.llm.provider_types import Provider
from fast_agent.ui.adapters.tui_io import TuiCommandIO
from fast_agent.ui.model_picker import _model_availability_display, _SplitListPicker
from fast_agent.ui.model_picker_common import (
    ModelSource,
    ProviderActivation,
    build_snapshot,
    model_options_for_provider,
)


@pytest.mark.parametrize("source", ["curated", "all"])
@pytest.mark.parametrize("api_key", [False, True])
@pytest.mark.parametrize("oauth_state", ["missing", "expired", "ready"])
def test_fast_picker_requires_oauth(
    monkeypatch: pytest.MonkeyPatch, source: ModelSource, api_key: bool, oauth_state: str
) -> None:
    monkeypatch.setenv("XAI_API_KEY", "")
    monkeypatch.setattr(
        "fast_agent.llm.provider.openai.xai_oauth.get_xai_token_status",
        lambda: {"present": oauth_state != "missing", "expired": oauth_state == "expired"},
    )
    monkeypatch.setattr(
        "fast_agent.llm.provider.openai.xai_oauth.get_xai_access_token",
        lambda: "oauth-token" if oauth_state == "ready" else None,
    )
    payload = {"xai": {"api_key": "test-key"}} if api_key else {}
    snapshot = build_snapshot(config_payload=payload)
    options = model_options_for_provider(snapshot, Provider.XAI, source=source)
    fast = next(option for option in options if option.spec == "xai.grok-4.7-build-fast")
    ordinary = next(option for option in options if option.spec == "xai.grok-4.7")
    activation = ProviderActivation(Provider.XAI)
    assert fast.activation_action == (None if oauth_state == "ready" else activation)
    assert ordinary.activation_action == (None if api_key or oauth_state == "ready" else activation)
    provider = next(option for option in snapshot.providers if option.provider == Provider.XAI)
    assert _model_availability_display(fast, provider_available=provider.active).availability == (
        "active" if oauth_state == "ready" else "attention"
    )

    picker = _SplitListPicker(config_path=None, initial_provider="xai")
    picker.snapshot = snapshot
    picker.state.provider_index = snapshot.providers.index(provider)
    picker.state.source = source
    picker.state.model_index = options.index(fast)
    result = picker._selected_result()
    assert result is not None
    assert result.selected_model == fast.spec
    assert result.activation_action == fast.activation_action


@pytest.mark.asyncio
async def test_fast_activation_uses_existing_xai_login_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XAI_API_KEY", "api-key-does-not-replace-oauth")
    monkeypatch.setattr(
        "fast_agent.llm.provider.openai.xai_oauth.get_xai_token_status",
        lambda: {"present": False, "expired": False},
    )
    login = Mock(return_value=OAuthCredential(access_token="oauth-token"))
    monkeypatch.setattr("fast_agent.llm.provider.openai.xai_oauth.login_xai_oauth", login)
    monkeypatch.setattr("fast_agent.ui.console.ensure_blocking_console", lambda: None)
    action = ProviderActivation(Provider.XAI)
    assert activate_model_picker_provider(action)
    login.assert_called_once_with()

    login.reset_mock()
    io = TuiCommandIO(Mock(spec=AgentProvider), "alpha", config_payload={})
    assert await io._handle_model_activation(action)
    login.assert_called_once_with()
