"""Submodule-local tests for the shared identity check.

The parent jarvis repo carries the end-to-end contract tests; this file
covers the helper in isolation so a regression in fast-agent surfaces
at submodule CI time instead of only when the parent repo bumps the
pointer.

Contract (see ``_team_helpers.assert_self_identity``):

- ``claimed_name=""`` → auto-detect, no impersonation risk
- ``claimed_name`` set + ``TEAM_MY_NAME`` matches → allow
- ``claimed_name`` set + ``TEAM_MY_NAME`` mismatch → REFUSE
- ``claimed_name`` set + ``TEAM_MY_NAME`` unset → REFUSE

No permissive escape hatch.
"""
from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from fast_agent.spawn.servers._team_helpers import assert_self_identity


# ── Helper: drive the check with controlled env + return parsed shape ──


def _run(claimed: str, *, env_name: str = ""):
    """Invoke assert_self_identity with TEAM_MY_NAME set to env_name (or unset)."""
    overrides = {"TEAM_MY_NAME": env_name} if env_name else {}
    with patch.dict(os.environ, overrides, clear=False):
        if not env_name:
            os.environ.pop("TEAM_MY_NAME", None)
        # Also patch get_my_name so the auto-detect branch returns a
        # predictable value regardless of host env.
        with patch(
            "fast_agent.spawn.servers._team_helpers.get_my_name",
            return_value=env_name or "agent",
        ):
            return assert_self_identity(claimed)


# ── Allow path ──


def test_auto_detect_when_claim_empty():
    """Empty claim → no impersonation risk, fall back to get_my_name."""
    resolved, err = _run("", env_name="Cameron [PM]")
    assert err is None
    assert resolved == "Cameron [PM]"


def test_allow_self_call_with_matching_claim():
    """Caller claims their own name — allowed."""
    resolved, err = _run("Cameron [PM]", env_name="Cameron [PM]")
    assert err is None
    assert resolved == "Cameron [PM]"


def test_match_is_case_insensitive_and_strips_whitespace():
    """LLMs occasionally lowercase or pad role tags — accept either."""
    resolved, err = _run("  cameron [pm]  ", env_name="Cameron [PM]")
    assert err is None
    # Caller's own spelling preserved on return (just normalized for the
    # comparison itself).
    assert resolved == "  cameron [pm]  "


# ── Refuse path ──


def test_refuse_when_claim_mismatches_env():
    """Caller claims a teammate's name — REFUSE, return Impersonation error."""
    resolved, err = _run("Sawyer [BA]", env_name="Cameron [PM]")
    assert resolved == ""
    assert err is not None
    err_data = json.loads(err)
    assert "Impersonation refused" in err_data["error"]
    assert err_data["caller_env"] == "Cameron [PM]"
    assert err_data["claimed_agent_name"] == "Sawyer [BA]"


def test_refuse_when_env_unset_and_claim_made():
    """No ground truth + a claim → REFUSE. The old permissive branch
    that returned the claim as-is is what re-opened the 2026-05-20
    impersonation incident; this path is the closed hole."""
    resolved, err = _run("Sawyer [BA]", env_name="")
    assert resolved == ""
    assert err is not None
    err_data = json.loads(err)
    assert "Identity unverifiable" in err_data["error"]
    assert err_data["caller_env"] == ""
    assert err_data["claimed_agent_name"] == "Sawyer [BA]"


def test_refuse_pm_force_skip_pattern_all_six_blocked():
    """End-to-end replay: PM tries to write turns for all 6 teammates.
    Every claim must be refused — production 2026-05-20 incident replay."""
    pm_env = "Taylor [PM]"
    teammates = [
        "Sawyer [BA]",
        "Reagan [SA]",
        "Eden [Dev]",
        "Devon [Designer]",
        "Kai [QE]",
        "Kai [DSO]",
    ]
    blocked = 0
    for mate in teammates:
        _, err = _run(mate, env_name=pm_env)
        if err and "Impersonation refused" in json.loads(err)["error"]:
            blocked += 1
    assert blocked == len(teammates), (
        f"Expected all {len(teammates)} impersonation attempts to be "
        f"blocked; only {blocked} were."
    )


# ── Error-shape consistency ──
#
# Both refuse paths (mismatch + unset-env) must return the same JSON
# shape so downstream parsers can read one set of keys regardless of
# which branch fired. The fast-agent#5 review caller flagged that the
# old implementation used "caller" in one path and dropped the field
# entirely in the other.


def test_refuse_shapes_share_keys():
    _, err_mismatch = _run("Sawyer [BA]", env_name="Cameron [PM]")
    _, err_unset = _run("Sawyer [BA]", env_name="")
    assert err_mismatch is not None and err_unset is not None
    keys_mismatch = set(json.loads(err_mismatch).keys())
    keys_unset = set(json.loads(err_unset).keys())
    assert keys_mismatch == keys_unset == {
        "error", "caller_env", "claimed_agent_name",
    }
