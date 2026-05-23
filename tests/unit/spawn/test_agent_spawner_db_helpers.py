"""Tests for the DB persistence helpers in agent_spawner_server.

These are the pieces the MCP subprocess uses to converge with the
parent project's ``agent_definitions`` table — schema mirrored on this
side (the submodule must not import the parent). The biggest
correctness contract: a `spawn_agent` followed by a
`remove_spawned_agent` MUST end with no row in the DB. Otherwise the
parent's poll loop re-attaches the "deleted" agent on its next tick.
"""

from __future__ import annotations

import sqlite3

import pytest

from fast_agent.spawn.servers.agent_spawner_server import (
    _delete_dynamic_agent_from_db,
    _open_dynamic_agents_db,
    _persist_dynamic_agent_to_db,
)


@pytest.fixture()
def db_path(tmp_path):
    """Empty SQLite file; tables are lazily created by the first
    helper call."""
    return str(tmp_path / "test_spawner.db")


def _rev(db_path: str) -> int:
    conn = _open_dynamic_agents_db(db_path)
    try:
        row = conn.execute(
            "SELECT value FROM agent_definitions_meta WHERE key = 'rev'"
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        conn.close()


# ── _persist_dynamic_agent_to_db ─────────────────────────────────────


def test_persist_creates_row_and_bumps_rev(db_path):
    _persist_dynamic_agent_to_db(
        db_path=db_path,
        name="Researcher",
        instruction="Find things.",
        servers=["serpapi", "scrapling-server"],
        model="anthropic.claude-sonnet",
    )

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT name, instruction, servers, model FROM agent_definitions WHERE name = ?",
            ("Researcher",),
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row[0] == "Researcher"
    assert row[1] == "Find things."
    assert row[2] == '["serpapi", "scrapling-server"]'
    assert row[3] == "anthropic.claude-sonnet"
    assert _rev(db_path) == 1


def test_persist_duplicate_name_raises_value_error(db_path):
    """Duplicate-name contract must match the parent's service-layer
    ``create_definition`` — both raise ValueError so the MCP tool can
    return a structured error to Jarvis."""
    _persist_dynamic_agent_to_db(
        db_path=db_path, name="Dup", instruction="first", servers=[], model=None
    )

    with pytest.raises(ValueError, match="already exists"):
        _persist_dynamic_agent_to_db(
            db_path=db_path, name="Dup", instruction="second", servers=[], model=None
        )


def test_persist_does_not_bump_rev_on_failure(db_path):
    """Failed persist must not bump rev — otherwise the parent's
    poll loop spins on a phantom change."""
    _persist_dynamic_agent_to_db(
        db_path=db_path, name="X", instruction="x", servers=[], model=None
    )
    rev_before = _rev(db_path)

    with pytest.raises(ValueError):
        _persist_dynamic_agent_to_db(
            db_path=db_path, name="X", instruction="dup", servers=[], model=None
        )
    assert _rev(db_path) == rev_before


def test_persist_null_model_stored_as_null(db_path):
    _persist_dynamic_agent_to_db(
        db_path=db_path, name="NoModel", instruction="x", servers=[], model=None
    )
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT model FROM agent_definitions WHERE name = ?", ("NoModel",)
        ).fetchone()
    finally:
        conn.close()
    assert row[0] is None


# ── _delete_dynamic_agent_from_db ────────────────────────────────────


def test_delete_drops_row_and_bumps_rev(db_path):
    """The bug this guards (PR #2 review finding): remove_spawned_agent
    must actually delete the agent_definitions row, not just remove
    a (non-existent) `.md` file or registry row.
    """
    _persist_dynamic_agent_to_db(
        db_path=db_path, name="ToRemove", instruction="x", servers=[], model=None
    )
    rev_after_create = _rev(db_path)

    removed = _delete_dynamic_agent_from_db(db_path=db_path, name="ToRemove")

    assert removed is True
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM agent_definitions WHERE name = ?", ("ToRemove",)
        ).fetchone()
    finally:
        conn.close()
    assert row is None
    assert _rev(db_path) == rev_after_create + 1


def test_delete_missing_is_idempotent_and_no_rev_bump(db_path):
    """Deleting a nonexistent name returns False without bumping rev
    so the caller can safely retry without triggering a useless
    reload on the parent."""
    _persist_dynamic_agent_to_db(
        db_path=db_path, name="Keep", instruction="x", servers=[], model=None
    )
    rev_before = _rev(db_path)

    assert _delete_dynamic_agent_from_db(db_path=db_path, name="NoSuch") is False
    assert _rev(db_path) == rev_before


# ── Round-trip: spawn → remove leaves a clean DB ─────────────────────


def test_spawn_then_remove_leaves_no_row_and_rev_bumped_twice(db_path):
    """End-to-end correctness against the PR #2 review BUG: after a
    spawn followed by a remove the DB must be empty, and the rev
    counter must have advanced exactly twice (once per mutation) so
    the parent's poll loop sees both events."""
    rev0 = _rev(db_path)

    _persist_dynamic_agent_to_db(
        db_path=db_path,
        name="EphemeralAgent",
        instruction="brief life",
        servers=["s1"],
        model=None,
    )
    rev1 = _rev(db_path)
    assert rev1 == rev0 + 1

    assert _delete_dynamic_agent_from_db(
        db_path=db_path, name="EphemeralAgent"
    ) is True
    rev2 = _rev(db_path)
    assert rev2 == rev1 + 1

    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute(
            "SELECT COUNT(*) FROM agent_definitions"
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 0
