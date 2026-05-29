"""Regression tests for the auto-resume team-identity cascade fix.

Incident reference: 2026-05-17 notification #28 ("Team ... hoàn thành (16 agents)")
fired while PM was actively mid-turn. Root cause traced to
``_check_and_resume_on_inbox`` not forwarding ``team_name`` + ``session_id``
to ``run_isolated_agent_background`` — each auto-resume cleared the team
identity from the new run's registry record, making auto-resumed workers
invisible to ``spawn_progress_bridge.find_by_team_name(...) ∩ session_id``.

These tests pin the recovery so the cascade can't silently come back.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from fast_agent.spawn import isolated_spawner
from fast_agent.spawn.spawn_registry import SpawnRecord


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def fake_messages_dir(tmp_path: Path) -> Path:
    """Create an empty messages dir so the path-resolution branch passes."""
    d = tmp_path / ".runtime" / "state" / "messages" / "sess123"
    d.mkdir(parents=True)
    return d


def _make_record(
    *,
    run_id: str = "run_fresh",
    team_name: str = "agile-team",
    session_id: str = "sess123",
    cfg_team_name: str | None = "agile-team",
    cfg_session_id_env: str | None = "sess123",
    messages_dir: Path | None = None,
) -> SpawnRecord:
    """Build a SpawnRecord with toggleable team identity at each layer."""
    env_vars: dict[str, str] = {}
    if cfg_session_id_env is not None:
        env_vars["TEAM_SESSION_ID"] = cfg_session_id_env
    if messages_dir is not None:
        env_vars["TEAM_MESSAGES_DIR"] = str(messages_dir)
    cfg: dict = {
        "task": "old task",
        "instruction": "old inst",
        "context": "old ctx",
        "servers": ["filesystem"],
        "model": "",
        "timeout_seconds": 0,
        "role": "dev",
        "workspace_dir": "",
        "project_dir": "/tmp/proj",
        "env_vars": env_vars,
    }
    if cfg_team_name is not None:
        cfg["team_name"] = cfg_team_name
    return SpawnRecord(
        run_id=run_id,
        agent_name="Phoenix [Dev]",
        role="dev",
        team_name=team_name,
        session_id=session_id,
        status="idle",
        original_config=cfg,
        result="prev result",
    )


@pytest.fixture
def mock_message_bus(monkeypatch):
    """MessageBus.read_unread returns ONE unread message so resume fires."""
    unread = [
        SimpleNamespace(
            from_name="PM", message_type="email", message_id="m1",
            content="please continue",
        ),
    ]
    bus_instance = MagicMock()
    bus_instance.read_unread.return_value = unread
    bus_instance.mark_all_done.return_value = None
    monkeypatch.setattr(
        isolated_spawner, "MessageBus" if hasattr(isolated_spawner, "MessageBus") else "",
        MagicMock(return_value=bus_instance),
        raising=False,
    )
    # ``_check_and_resume_on_inbox`` imports MessageBus inside the function;
    # patch the module it imports from.
    import fast_agent.spawn.message_bus as mb_mod
    monkeypatch.setattr(mb_mod, "MessageBus", MagicMock(return_value=bus_instance))
    return bus_instance


@pytest.fixture
def captured_resume(monkeypatch):
    """Replace ``run_isolated_agent_background`` with an AsyncMock that
    records the kwargs of the resume call.
    """
    mock = AsyncMock(return_value="new_run_42")
    monkeypatch.setattr(
        isolated_spawner, "run_isolated_agent_background", mock,
    )
    return mock


@pytest.fixture
def fake_registry():
    """Minimal registry stub with .get() / has_running_resume() / mutation."""
    reg = MagicMock()
    reg.has_running_resume.return_value = False
    reg._load = MagicMock()
    reg._save = MagicMock()
    reg._data = {}
    return reg


# ── Tests ────────────────────────────────────────────────────────────


@pytest.mark.anyio
async def test_resume_forwards_team_identity_when_fresh(
    fake_registry, mock_message_bus, captured_resume, fake_messages_dir,
):
    """Fresh record (cfg.team_name + record.team_name both set) → resume
    call MUST receive team_name and session_id, preventing the next link
    from drifting.
    """
    record = _make_record(messages_dir=fake_messages_dir)
    fake_registry.get.return_value = record
    fake_registry._data[record.run_id] = {"restart_count": 0}

    await isolated_spawner._check_and_resume_on_inbox(
        run_id=record.run_id,
        agent_name=record.agent_name,
        registry=fake_registry,
        env_vars=record.original_config["env_vars"],
    )

    captured_resume.assert_called_once()
    kwargs = captured_resume.call_args.kwargs
    assert kwargs.get("team_name") == "agile-team", (
        "Fresh record: cfg.team_name must propagate to the new run."
    )
    assert kwargs.get("session_id") == "sess123", (
        "Fresh record: env.TEAM_SESSION_ID must propagate to the new run."
    )


@pytest.mark.anyio
async def test_resume_recovers_team_name_via_db_lookup_when_drifted(
    fake_registry, mock_message_bus, captured_resume, fake_messages_dir,
    monkeypatch,
):
    """Drifted record (cfg.team_name="" + record.team_name="") but
    env.TEAM_SESSION_ID still present → MUST recover team_name by
    looking up team_sessions[session_id].team_name and pass it along.
    """
    record = _make_record(
        team_name="",
        cfg_team_name="",
        messages_dir=fake_messages_dir,
    )
    fake_registry.get.return_value = record
    fake_registry._data[record.run_id] = {"restart_count": 0}

    # Mock team_spawner.get_team_session(session_id) → returns object with .team_name
    fake_session = SimpleNamespace(team_name="agile-team")
    import fast_agent.spawn.team_spawner as ts_mod
    monkeypatch.setattr(
        ts_mod, "get_team_session", lambda sid: fake_session,
    )

    await isolated_spawner._check_and_resume_on_inbox(
        run_id=record.run_id,
        agent_name=record.agent_name,
        registry=fake_registry,
        env_vars=record.original_config["env_vars"],
    )

    captured_resume.assert_called_once()
    kwargs = captured_resume.call_args.kwargs
    assert kwargs.get("team_name") == "agile-team", (
        "DB recovery via team_sessions[session_id] must restore team_name "
        "when both cfg and record layers have drifted to empty."
    )
    assert kwargs.get("session_id") == "sess123"


@pytest.mark.anyio
async def test_resume_warns_loudly_when_team_name_unrecoverable(
    fake_registry, mock_message_bus, captured_resume, fake_messages_dir,
    monkeypatch, caplog,
):
    """All recovery sources empty → still proceed (resume must not be
    blocked) but emit a WARNING so an operator can investigate.
    """
    record = _make_record(
        team_name="",
        session_id="",
        cfg_team_name="",
        cfg_session_id_env=None,
        messages_dir=fake_messages_dir,
    )
    fake_registry.get.return_value = record
    fake_registry._data[record.run_id] = {"restart_count": 0}

    # Capture warnings DIRECTLY off the module-level ``logger`` reference
    # rather than via pytest's ``caplog`` fixture. Other test modules
    # (notably ``tests/unit/hf_inference_acp/test_wizard_curated_models``
    # which boots WizardSetupLLM) import the full fast_agent runtime stack
    # and reconfigure the ``fast_agent.spawn`` logger hierarchy in ways
    # that caplog handlers can't reliably hook into mid-session.
    # Patching the symbol the call-site actually uses (``isolated_spawner.logger``)
    # bypasses the global logging plumbing entirely and pins the behaviour
    # we care about: did the code path execute its WARNING emission?
    warnings_seen: list[str] = []
    real_logger = isolated_spawner.logger

    class _CaptureLogger:
        def __getattr__(self, name):
            if name == "warning":
                def _wrap(msg, *args, **kwargs):
                    try:
                        warnings_seen.append(msg % args if args else msg)
                    except Exception:
                        warnings_seen.append(str(msg))
                    return real_logger.warning(msg, *args, **kwargs)
                return _wrap
            return getattr(real_logger, name)

    monkeypatch.setattr(isolated_spawner, "logger", _CaptureLogger())
    await isolated_spawner._check_and_resume_on_inbox(
        run_id=record.run_id,
        agent_name=record.agent_name,
        registry=fake_registry,
        env_vars=record.original_config["env_vars"],
    )

    assert any(
        "missing session_id" in m or "missing team_name" in m
        or ("session_id=" in m and "team_name=" in m)
        for m in warnings_seen
    ), (
        "Must log a WARNING when team identity is unrecoverable. "
        f"Captured warnings: {warnings_seen!r}"
    )


@pytest.mark.anyio
async def test_resume_uses_record_session_id_when_env_missing(
    fake_registry, mock_message_bus, captured_resume, fake_messages_dir,
):
    """env.TEAM_SESSION_ID missing but record.session_id populated → must
    fall back to the top-level record field.
    """
    record = _make_record(
        cfg_session_id_env=None,
        messages_dir=fake_messages_dir,
    )
    fake_registry.get.return_value = record
    fake_registry._data[record.run_id] = {"restart_count": 0}

    await isolated_spawner._check_and_resume_on_inbox(
        run_id=record.run_id,
        agent_name=record.agent_name,
        registry=fake_registry,
        env_vars=record.original_config["env_vars"],
    )

    captured_resume.assert_called_once()
    kwargs = captured_resume.call_args.kwargs
    assert kwargs.get("session_id") == "sess123", (
        "When env.TEAM_SESSION_ID is missing, fall back to record.session_id."
    )


# ── server_overrides + workspace_dir preservation ──────────────────


@pytest.mark.anyio
async def test_resume_forwards_server_overrides_when_fresh(
    fake_registry, mock_message_bus, captured_resume, fake_messages_dir,
):
    """Fresh cfg has server_overrides → resume call MUST receive it.

    Without this, filesystem MCP falls back to base config defaults; the
    historical default (``./jarvis_workspace``) pointed at a non-existent
    dir → MCP failed to start → agent saw no filesystem tool (incident
    2026-05-17 Designer).
    """
    record = _make_record(messages_dir=fake_messages_dir)
    record.original_config["server_overrides"] = {
        "filesystem": {
            "args": ["-y", "@modelcontextprotocol/server-filesystem",
                     "{workspace_dir}", "{project_dir}/.fast-agent/skills"],
        }
    }
    record.original_config["workspace_dir"] = "/tmp/team_ws"
    fake_registry.get.return_value = record
    fake_registry._data[record.run_id] = {"restart_count": 0}

    await isolated_spawner._check_and_resume_on_inbox(
        run_id=record.run_id,
        agent_name=record.agent_name,
        registry=fake_registry,
        env_vars=record.original_config["env_vars"],
    )

    captured_resume.assert_called_once()
    kwargs = captured_resume.call_args.kwargs
    assert kwargs.get("server_overrides") == record.original_config["server_overrides"], (
        "server_overrides MUST be forwarded so per-role filesystem args survive resume"
    )
    assert kwargs.get("workspace_dir") == "/tmp/team_ws", (
        "workspace_dir MUST be forwarded so {workspace_dir} placeholder resolves"
    )


@pytest.mark.anyio
async def test_resume_recovers_server_overrides_via_db_lookup(
    fake_registry, mock_message_bus, captured_resume, fake_messages_dir,
    monkeypatch,
):
    """Drifted cfg (server_overrides=None) + session_id known →
    MUST recover server_overrides from team_sessions.template.roles[role].
    """
    record = _make_record(messages_dir=fake_messages_dir)
    # Simulate cascade: server_overrides dropped through prior resumes
    record.original_config["server_overrides"] = None
    record.original_config["workspace_dir"] = ""
    fake_registry.get.return_value = record
    fake_registry._data[record.run_id] = {"restart_count": 0}

    # DB-side SSoT: team_sessions has the canonical per-role override
    canonical_overrides = {
        "filesystem": {
            "args": ["-y", "@modelcontextprotocol/server-filesystem",
                     "{workspace_dir}", "{project_dir}/.fast-agent/skills"],
        }
    }
    fake_session = SimpleNamespace(
        team_name="agile-team",
        workspace="/tmp/team_ws_from_session",
        template={
            "roles": {
                "dev": {"server_overrides": canonical_overrides},
            },
        },
    )
    import fast_agent.spawn.team_spawner as ts_mod
    monkeypatch.setattr(
        ts_mod, "get_team_session", lambda sid: fake_session,
    )

    await isolated_spawner._check_and_resume_on_inbox(
        run_id=record.run_id,
        agent_name=record.agent_name,
        registry=fake_registry,
        env_vars=record.original_config["env_vars"],
    )

    captured_resume.assert_called_once()
    kwargs = captured_resume.call_args.kwargs
    assert kwargs.get("server_overrides") == canonical_overrides, (
        "DB recovery via team_sessions.template.roles[role].server_overrides "
        "must restore the per-role overrides lost through the cascade."
    )
    assert kwargs.get("workspace_dir") == "/tmp/team_ws_from_session", (
        "workspace_dir also recovered from session.workspace when cfg drifted."
    )


# ── Regression: original_config persists server_overrides at register time ──
#
# Without this, even a fresh spawn loses server_overrides on the FIRST
# auto-resume because cfg.get("server_overrides") returns None — there's
# nothing to forward. This test pins the persistence at the registration
# site so the field is always readable for subsequent resumes.


@pytest.mark.anyio
async def test_register_persists_server_overrides_in_original_config(
    monkeypatch,
):
    """End-to-end registry check: a fresh spawn with server_overrides MUST
    write them into record.original_config — otherwise auto-resume has
    nothing to read back.
    """
    from fast_agent.spawn.spawn_registry import SpawnRegistry

    captured_record: dict = {}
    fake_registry = MagicMock(spec=SpawnRegistry)
    fake_registry.register = MagicMock(
        side_effect=lambda rec: captured_record.setdefault("rec", rec),
    )
    fake_registry.update_status = MagicMock()

    # Block the actual subprocess; we only care about the register() call.
    async def _noop(*a, **kw):
        return {"status": "completed", "result": "", "metadata": {}}

    monkeypatch.setattr(isolated_spawner, "run_isolated_agent", _noop)

    overrides = {
        "filesystem": {
            "args": ["-y", "@modelcontextprotocol/server-filesystem",
                     "{workspace_dir}"],
        }
    }
    await isolated_spawner.run_isolated_agent_background(
        task="t", project_dir="/tmp", instruction="i", context="c",
        servers=["filesystem"], role="dev", agent_name="X",
        team_name="agile-team", lifecycle="resumable",
        registry=fake_registry, server_overrides=overrides,
        session_id="sess123", workspace_dir="/tmp/ws",
    )

    # Wait for background task to register
    import asyncio
    await asyncio.sleep(0.1)

    rec = captured_record.get("rec")
    assert rec is not None, "registry.register MUST have been called"
    cfg = rec.original_config
    assert cfg.get("server_overrides") == overrides, (
        "server_overrides MUST be persisted into original_config so the "
        "next resume can read it back. This is the single source of truth "
        "for per-role MCP arg customizations after the cascade fix."
    )
    assert cfg.get("workspace_dir") == "/tmp/ws", (
        "workspace_dir MUST be in original_config so resume preserves it"
    )
