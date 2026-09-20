import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from fast_agent.cli.runtime import copilot_activation as activation
from fast_agent.config import CopilotSettings
from fast_agent.core.exceptions import ProviderKeyError
from fast_agent.llm.provider.copilot.models import COPILOT_MODELS
from fast_agent.llm.provider_types import Provider
from fast_agent.ui.model_picker_common import (
    build_snapshot,
    model_options_for_provider,
    provider_credential_summary,
)


@pytest.mark.parametrize("source", ["curated", "all"])
def test_picker_contract(source):
    snapshot = build_snapshot(config_payload={})
    options = model_options_for_provider(snapshot, Provider.COPILOT, source=source)
    assert [option.spec for option in options] == [f"copilot.{key}" for key in COPILOT_MODELS]
    assert all(option.activation_action is not None for option in options)
    summary = provider_credential_summary(Provider.COPILOT, {})
    assert not summary.active
    assert summary.label is None


def _broker_factory(monkeypatch, module, broker):
    """Stand in for the broker constructor, recording requested settings."""
    factory = Mock(return_value=broker)
    monkeypatch.setattr(module, "CopilotBroker", factory)
    return factory


@pytest.fixture(autouse=True)
def isolated_auth(monkeypatch):
    monkeypatch.delenv("COPILOT_GITHUB_TOKEN", raising=False)
    forbidden = Mock(side_effect=AssertionError("No executable or subprocess allowed"))
    monkeypatch.setattr("shutil.which", forbidden)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", forbidden)


@pytest.fixture
def auth(monkeypatch):
    broker = Mock(has_credentials=AsyncMock())
    _broker_factory(monkeypatch, activation, broker)
    confirm = Mock(return_value=True)
    monkeypatch.setattr(activation.typer, "confirm", confirm)
    login = AsyncMock()
    monkeypatch.setattr(activation, "login_copilot_oauth_async", login)
    monkeypatch.setattr("fast_agent.ui.console.ensure_blocking_console", Mock())
    return broker, login, confirm


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "statuses,expected,login_expected",
    [([True], True, False), ([False, True], True, True), ([False, False], False, True)],
)
async def test_activate(auth, statuses, expected, login_expected):
    broker, login, confirm = auth
    broker.has_credentials.side_effect = statuses
    assert await activation.activate_copilot(CopilotSettings()) is expected
    assert login.called is login_expected
    assert confirm.called is login_expected
    assert broker.has_credentials.await_count == len(statuses)


@pytest.mark.asyncio
async def test_consent_before_login_and_recheck(auth, monkeypatch):
    broker, login, confirm = auth
    events = Mock()
    blocking = Mock()
    monkeypatch.setattr("fast_agent.ui.console.ensure_blocking_console", blocking)
    for name, mock in [
        ("check", broker.has_credentials),
        ("blocking", blocking),
        ("confirm", confirm),
        ("login", login),
    ]:
        events.attach_mock(mock, name)
    broker.has_credentials.side_effect = [False, True]
    assert await activation.activate_copilot(CopilotSettings())
    assert [call[0] for call in events.mock_calls] == [
        "check",
        "blocking",
        "confirm",
        "login",
        "check",
    ]
    assert confirm.call_args.kwargs["default"] is False


@pytest.mark.asyncio
async def test_decline(auth):
    broker, login, confirm = auth
    broker.has_credentials.return_value = False
    confirm.return_value = False
    assert not await activation.activate_copilot(CopilotSettings())
    login.assert_not_called()
    broker.has_credentials.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("token", ["", "explicit-token"])
@pytest.mark.parametrize("authenticated", [False, True])
async def test_environment_never_logs_in(auth, monkeypatch, token, authenticated):
    broker, login, confirm = auth
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", token)
    broker.has_credentials.return_value = authenticated
    assert await activation.activate_copilot(CopilotSettings()) is authenticated
    login.assert_not_called()
    confirm.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["check", "login", "recheck"])
async def test_auth_failure_retains_help(auth, monkeypatch, capsys, stage):
    broker, login, confirm = auth
    error = ProviderKeyError("Authentication failed", "Detailed recovery help.\nRetry selection.")
    broker.has_credentials.side_effect = [False, True]
    if stage == "check":
        broker.has_credentials.side_effect = error
    elif stage == "login":
        login.side_effect = error
    else:
        broker.has_credentials.side_effect = [False, error]
    assert not await activation.activate_copilot(CopilotSettings())
    assert (
        "Authentication failed: Detailed recovery help. Retry selection." in capsys.readouterr().err
    )
    if stage == "check":
        login.assert_not_called()
        confirm.assert_not_called()
    if stage == "login":
        broker.has_credentials.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [EOFError, KeyboardInterrupt, activation.typer.Abort])
@pytest.mark.parametrize("stage", ["confirm", "login"])
async def test_interactive_cancellation(auth, error, stage):
    broker, login, confirm = auth
    broker.has_credentials.return_value = False
    (confirm if stage == "confirm" else login).side_effect = error
    assert not await activation.activate_copilot(CopilotSettings())
    broker.has_credentials.assert_awaited_once()
    if stage == "confirm":
        login.assert_not_called()


@pytest.mark.asyncio
async def test_task_cancellation_stops_login(auth):
    broker, login, _ = auth
    broker.has_credentials.return_value = False
    started = asyncio.Event()
    stopped = asyncio.Event()
    saved = Mock()

    async def poll():
        started.set()
        try:
            await asyncio.Event().wait()
            saved()
        finally:
            stopped.set()

    login.side_effect = poll
    task = asyncio.create_task(activation.activate_copilot(CopilotSettings()))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped.is_set()
    saved.assert_not_called()
    broker.has_credentials.assert_awaited_once()


async def _settle_preflight(picker_instance) -> None:
    """Wait for the background Copilot probe and its done-callback to run."""
    task = picker_instance._copilot_preflight
    assert task is not None
    await task
    await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [True, False, "check_failure"])
async def test_picker_open_preflight(monkeypatch, status):
    from fast_agent.llm.provider.copilot import broker as broker_module
    from fast_agent.ui import model_picker as picker

    broker = Mock(has_credentials=AsyncMock(return_value=status is True))
    if status == "check_failure":
        broker.has_credentials.side_effect = ProviderKeyError("failed")
    factory = _broker_factory(monkeypatch, broker_module, broker)
    login = AsyncMock(side_effect=AssertionError("Preflight must not log in"))
    monkeypatch.setattr(activation, "login_copilot_oauth_async", login)
    monkeypatch.setattr("fast_agent.llm.provider.copilot.oauth.login_copilot_oauth_async", login)
    snapshots = []

    async def run(self):
        await _settle_preflight(self)
        snapshots.append(self.snapshot)
        return None

    monkeypatch.setattr(picker._SplitListPicker, "run_async", run)
    await picker.run_model_picker_async(
        config_payload={"copilot": {"base_url": "https://copilot.invalid"}}
    )
    snapshot = snapshots[0]
    option = next(o for o in snapshot.providers if o.provider == Provider.COPILOT)
    assert option.active is (status is True)
    assert option.credential_label == ("OAuth" if status is True else None)
    models = model_options_for_provider(snapshot, Provider.COPILOT, source="curated")
    assert all((model.activation_action is None) is (status is True) for model in models)
    broker.has_credentials.assert_awaited_once_with(timeout=2.0)
    login.assert_not_called()
    assert factory.call_args.args[0].base_url == "https://copilot.invalid"
    broker.resolve.assert_not_called()


@pytest.mark.asyncio
async def test_preflight_loaded_settings_refresh(monkeypatch, tmp_path):
    from fast_agent.config import Settings
    from fast_agent.llm.provider.copilot import broker as broker_module
    from fast_agent.ui import model_picker as picker

    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "explicit-token")
    settings = Settings(
        copilot=CopilotSettings(base_url="https://copilot.invalid", runtime_timeout_seconds=0.01)
    )
    load = Mock(return_value=settings)
    monkeypatch.setattr("fast_agent.config.get_settings", load)
    broker = Mock(has_credentials=AsyncMock(side_effect=[True, False]))
    factory = _broker_factory(monkeypatch, broker_module, broker)
    options = []

    async def run(self):
        await _settle_preflight(self)
        options.append(next(o for o in self.snapshot.providers if o.provider == Provider.COPILOT))
        return None

    monkeypatch.setattr(picker._SplitListPicker, "run_async", run)
    path = tmp_path / "config.yaml"
    for _ in range(2):
        await picker.run_model_picker_async(config_path=path)
    load.assert_called_with(str(path))
    assert factory.call_args.args[0] == settings.copilot
    assert options[0].active and options[0].credential_label == "env"
    assert not options[1].active


@pytest.mark.asyncio
async def test_picker_opens_before_preflight_and_cancels_on_exit(monkeypatch):
    from fast_agent.llm.provider.copilot import broker as broker_module
    from fast_agent.ui import model_picker as picker

    started = asyncio.Event()

    async def slow_check(timeout=None):
        started.set()
        await asyncio.sleep(60)
        return True

    _broker_factory(monkeypatch, broker_module, Mock(has_credentials=slow_check))

    pickers = []
    original_run_async = picker._SplitListPicker.run_async

    async def run(self):
        pickers.append(self)
        return await original_run_async(self)

    async def app_run(_app):
        # Simulate the user exiting while the Copilot probe is still pending.
        await started.wait()
        return None

    monkeypatch.setattr(picker._SplitListPicker, "run_async", run)
    monkeypatch.setattr(picker.Application, "run_async", app_run)

    assert await picker.run_model_picker_async(config_payload={}) is None

    (instance,) = pickers
    option = next(o for o in instance.snapshot.providers if o.provider == Provider.COPILOT)
    assert option.active is False  # probe never resolved; activation flow remains
    with pytest.raises(asyncio.CancelledError):
        await instance._copilot_preflight


@pytest.mark.parametrize("authenticated", [False, True])
@pytest.mark.parametrize("token", [None, "", "explicit-token"])
def test_rendering_never_reads_copilot_credentials(monkeypatch, authenticated, token):
    from fast_agent.llm.provider.copilot import oauth

    if token is not None:
        monkeypatch.setenv("COPILOT_GITHUB_TOKEN", token)
    forbidden = Mock(side_effect=AssertionError("Rendering must not read credentials"))
    for name in ["get_copilot_access_token", "get_copilot_token_status", "load_oauth_credential"]:
        monkeypatch.setattr(oauth, name, forbidden)
    summary = provider_credential_summary(Provider.COPILOT, {}, copilot_authenticated=authenticated)
    assert summary.active is authenticated
    assert summary.label == (("env" if token is not None else "OAuth") if authenticated else None)
    snapshot = build_snapshot(config_payload={}, copilot_authenticated=authenticated)
    option = next(o for o in snapshot.providers if o.provider == Provider.COPILOT)
    assert option.credential_label == summary.label
    forbidden.assert_not_called()


@pytest.mark.asyncio
async def test_failed_activation_can_retry(auth):
    broker, login, _ = auth
    broker.has_credentials.side_effect = [False, False, False, True]
    assert not await activation.activate_copilot(CopilotSettings())
    assert await activation.activate_copilot(CopilotSettings())
    assert login.await_count == 2
