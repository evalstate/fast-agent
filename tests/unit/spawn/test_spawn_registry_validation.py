"""Tests for ``SpawnRegistry.register`` write-side validation.

Pin the invariant that surfaced as the 2026-05-17 incident: a team-managed
agent must NEVER be registered under a non-distinct identity (empty
agent_name or agent_name equal to role). Without this guard, the dashboard
silently rendered two cards for the same logical PM ("Robin [PM]" + "pm").

Ad-hoc spawns (no ``team_name``) keep the legitimate fallback where
``agent_name=role`` is meaningful — those tests pin that we did NOT
over-tighten and break the generic ``_spawn_agent_background`` MCP tool.
"""
from __future__ import annotations

import pytest

from fast_agent.spawn.spawn_registry import (
    Lifecycle,
    SpawnRecord,
    SpawnRegistry,
    SpawnStatus,
)


def _make_record(**overrides) -> SpawnRecord:
    base = dict(
        run_id="run-test-123",
        agent_name="Robin [PM]",
        role="pm",
        team_name="agile-team",
        task="x",
        status=SpawnStatus.RUNNING.value,
        lifecycle=Lifecycle.RESUMABLE.value,
    )
    base.update(overrides)
    return SpawnRecord(**base)


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """Construct a SpawnRegistry pointed at a temp JSON file so writes are isolated.

    Forces the file backend (rather than the SQLite shared one) by NOT
    setting SPAWN_REGISTRY_DB and passing a .json registry_file. Keeps
    tests independent + parallel-safe.
    """
    monkeypatch.delenv("SPAWN_REGISTRY_DB", raising=False)
    return SpawnRegistry(registry_file=tmp_path / "spawn_registry.json")


# ── Validator: team-managed spawns ─────────────────────────────────────────


class TestTeamManagedValidation:
    def test_accepts_distinct_agent_name(self, registry):
        registry.register(_make_record(agent_name="Robin [PM]", role="pm", team_name="alpha"))
        assert registry.get("run-test-123") is not None

    def test_rejects_empty_agent_name_when_team_set(self, registry):
        with pytest.raises(ValueError, match="without agent_name"):
            registry.register(_make_record(agent_name="", role="pm", team_name="alpha"))

    def test_rejects_agent_name_equal_to_role(self, registry):
        with pytest.raises(ValueError, match="equals role|equals\\s+role"):
            registry.register(_make_record(agent_name="pm", role="pm", team_name="alpha"))

    def test_error_message_mentions_caller_responsibility(self, registry):
        """Error must point operator at the caller — not invite a 'just relax
        the validator' workaround. Pin the language so future contributors
        read it the same way.
        """
        with pytest.raises(ValueError) as exc_info:
            registry.register(_make_record(agent_name="pm", role="pm", team_name="alpha"))
        msg = str(exc_info.value)
        assert "Fix the caller" in msg or "do not loosen" in msg


# ── Validator: ad-hoc spawns preserved ─────────────────────────────────────


class TestAdHocPreservation:
    """The fallback ``agent_name=role`` is LEGITIMATE for ad-hoc spawns
    (no team_name). The ``_spawn_agent_background`` MCP tool uses this path
    when an orchestrator spawns a one-off internal helper. The validator
    MUST NOT break it.
    """

    def test_accepts_agent_name_equal_to_role_when_no_team(self, registry):
        registry.register(_make_record(agent_name="researcher", role="researcher", team_name=""))
        rec = registry.get("run-test-123")
        assert rec is not None and rec.agent_name == "researcher"

    def test_accepts_empty_agent_name_when_no_team(self, registry):
        # Top-level isolated_spawner uses ``agent_name or role or "agent"`` —
        # so an empty agent_name only reaches register() in pathological
        # cases. But the validator's domain is ONLY team-managed spawns,
        # so an ad-hoc empty agent_name must pass through.
        registry.register(_make_record(agent_name="", role="x", team_name=""))
        assert registry.get("run-test-123") is not None


# ── from_dict back-compat: pre-fix orphans must still load ─────────────────


class TestFromDictBackwardCompat:
    """The validator lives in ``register()`` — NOT ``__post_init__`` —
    specifically so existing rows in the DB that violate the new invariant
    (i.e. the 2026-05-17 orphan) can still be loaded for inspection /
    audit / cleanup. Without this, deploying the validator would crash on
    every read.
    """

    def test_from_dict_accepts_orphan_shape(self):
        # Mirror the production orphan exactly: team_name set, agent_name == role
        rec = SpawnRecord.from_dict({
            "run_id": "orphan-1",
            "agent_name": "pm",
            "role": "pm",
            "team_name": "agile-team",
            "status": "idle",
            "lifecycle": "resumable",
        })
        assert rec.run_id == "orphan-1"
        assert rec.agent_name == "pm"
        # No raise — we can still inspect / cleanup these via scripts.

    def test_round_trip_through_get_loads_orphan(self, registry, monkeypatch):
        """An orphan already in the backend (from before the validator
        landed) must still be readable via .get(). We bypass register()
        and write directly to the backend to simulate this state.
        """
        orphan = {
            "run_id": "orphan-1",
            "agent_name": "pm",
            "role": "pm",
            "team_name": "agile-team",
            "status": "idle",
            "lifecycle": "resumable",
        }
        # Inject directly via private backend — represents pre-validator data
        registry._data["orphan-1"] = orphan
        registry._save()
        loaded = registry.get("orphan-1")
        assert loaded is not None
        assert loaded.agent_name == "pm"


# ── Sanity: register actually writes ───────────────────────────────────────


def test_register_persists_to_backend(registry):
    """Smoke test — the validator must not accidentally swallow the write
    when the input IS valid. Belt + suspenders for the happy path.
    """
    registry.register(_make_record(run_id="happy-1"))
    rec = registry.get("happy-1")
    assert rec is not None
    assert rec.agent_name == "Robin [PM]"
    assert rec.role == "pm"
    assert rec.team_name == "agile-team"


# ── Lifecycle merge (2026-05-20): persistent → resumable ───────────────────


class TestLifecyclePersistentMerged:
    """The legacy ``"persistent"`` lifecycle was functionally identical to
    ``"resumable"`` everywhere except one place: ``isolated_spawner``
    write-side decided ``COMPLETED`` for persistent vs ``IDLE`` for
    resumable. We collapsed the two on 2026-05-20.

    These tests pin:
    1. The enum no longer ships ``PERSISTENT`` (compile-time guard).
    2. Reading a legacy DB row with ``lifecycle="persistent"`` upgrades it
       to ``"resumable"`` (back-compat without DB migration).
    3. Reading other lifecycle values is untouched (no false positives).
    """

    def test_enum_drops_persistent(self):
        # Old enum had PERSISTENT — make sure callers can't accidentally
        # construct it from the symbol again.
        assert not hasattr(Lifecycle, "PERSISTENT")
        assert {m.value for m in Lifecycle} == {"oneshot", "resumable"}

    def test_from_dict_coerces_legacy_persistent_to_resumable(self):
        rec = SpawnRecord.from_dict({
            "run_id": "legacy-persistent-1",
            "agent_name": "Sasha",
            "role": "agent",
            "lifecycle": "persistent",
            "status": "completed",
        })
        # Old "persistent" row reads back as "resumable" — downstream code
        # can rely on the 2-value enum without special-casing legacy data.
        assert rec.lifecycle == "resumable"

    def test_from_dict_preserves_oneshot(self):
        rec = SpawnRecord.from_dict({
            "run_id": "modern-oneshot-1",
            "agent_name": "Lex",
            "role": "agent",
            "lifecycle": "oneshot",
        })
        assert rec.lifecycle == "oneshot"

    def test_from_dict_preserves_resumable(self):
        rec = SpawnRecord.from_dict({
            "run_id": "modern-resumable-1",
            "agent_name": "Rey",
            "role": "agent",
            "lifecycle": "resumable",
        })
        assert rec.lifecycle == "resumable"

    def test_legacy_persistent_round_trip_via_registry(self, registry):
        """End-to-end: write a raw row with ``"persistent"`` directly into
        the backend (simulating a pre-merge DB), then read via .get() and
        verify the coerce kicks in at the boundary.
        """
        legacy_row = {
            "run_id": "legacy-end-to-end",
            "agent_name": "Sasha",
            "role": "agent",
            "team_name": "",
            "status": "completed",
            "lifecycle": "persistent",
        }
        registry._data["legacy-end-to-end"] = legacy_row
        registry._save()
        loaded = registry.get("legacy-end-to-end")
        assert loaded is not None
        assert loaded.lifecycle == "resumable", (
            "Legacy persistent record should be upgraded to resumable on read"
        )
