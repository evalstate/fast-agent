"""Tests for nested ``hc`` child sessions, terminal pointers, and usage rollup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest
from mcp.types import TextContent

from fast_agent.agents.agent_types import AgentConfig
from fast_agent.cli.runtime.hc_recovery import resolve_hc_parent_session
from fast_agent.cli.runtime.run_request import AgentRunRequest
from fast_agent.integrations.hc_children import HcSelfInfo
from fast_agent.llm.provider_types import Provider
from fast_agent.llm.usage_tracking import (
    CompletionTokenUsage,
    PromptTokenUsage,
    TurnUsage,
    UsageAccumulator,
    UsageSchema,
)
from fast_agent.session.session_manager import SessionManager
from fast_agent.session.snapshot import (
    SESSION_SNAPSHOT_SCHEMA_VERSION,
    SessionChildLinkSnapshot,
    load_session_snapshot,
)
from fast_agent.session.subagent_runs import (
    collect_hc_child_usage_rollups,
    subagent_run_from_session,
)
from fast_agent.session.terminal_pointers import (
    HcTerminalPointer,
    read_terminal_pointer,
    remove_terminal_pointer,
    write_terminal_pointer,
)
from fast_agent.types import PromptMessageExtended
from fast_agent.ui.usage_display import format_usage_markdown, usage_rows_from_hc_rollups

if TYPE_CHECKING:
    from pathlib import Path

    from fast_agent.interfaces import AgentProtocol


@dataclass
class _StubLlm:
    model_name: str = "test-model"
    provider: object = None
    default_request_params: object = None
    resolved_model: object = None


class _StubAgent:
    def __init__(
        self,
        name: str,
        *,
        model: str = "test-model",
        input_tokens: int = 100,
        output_tokens: int = 40,
        cache_read_tokens: int | None = 50,
        tool_calls: int = 1,
        cost_usd: float | None = 0.01,
    ) -> None:
        self.name = name
        self.instruction = "test"
        self.config = AgentConfig(name, model=model)
        self.llm = None
        self.message_history: list[PromptMessageExtended] = [
            PromptMessageExtended(role="user", content=[TextContent(type="text", text="hi")]),
            PromptMessageExtended(role="assistant", content=[TextContent(type="text", text="ok")]),
        ]
        self.usage_accumulator = UsageAccumulator()
        if input_tokens > 0 or output_tokens > 0:
            self.usage_accumulator.add_turn(
                TurnUsage(
                    provider=Provider.RESPONSES,
                    usage_schema=UsageSchema.OPENAI_RESPONSES,
                    model=model,
                    prompt=PromptTokenUsage(total=input_tokens, cache_read=cache_read_tokens),
                    completion=CompletionTokenUsage(total=output_tokens),
                    tool_calls=tool_calls,
                    cost_usd=cost_usd,
                )
            )
        self.subagent_usage_accumulator = UsageAccumulator()


def test_terminal_pointer_round_trip_and_safe_segments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    pointer = HcTerminalPointer(
        core_id="core-abc",
        terminal_id="t1",
        home_dir=tmp_path / ".fast-agent",
        session_id="2604141705-AbCd12",
        session_dir=tmp_path / ".fast-agent" / "sessions" / "2604141705-AbCd12",
        agent_name="main",
    )
    assert write_terminal_pointer(pointer) is not None
    loaded = read_terminal_pointer("core-abc", "t1")
    assert loaded is not None
    assert loaded.session_id == pointer.session_id
    assert loaded.agent_name == "main"

    # Reject path traversal segments
    assert read_terminal_pointer("../evil", "t1") is None
    assert read_terminal_pointer("core-abc", "desktop/t1") is None

    # Mismatched expected_session_id does not delete
    remove_terminal_pointer("core-abc", "t1", expected_session_id="other-session")
    assert read_terminal_pointer("core-abc", "t1") is not None

    remove_terminal_pointer("core-abc", "t1", expected_session_id=pointer.session_id)
    assert read_terminal_pointer("core-abc", "t1") is None


@pytest.mark.asyncio
async def test_hc_child_session_lifecycle_deletion_protection_and_lock_free_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    home_dir = tmp_path / ".fast-agent"

    parent_mgr = SessionManager(cwd=tmp_path, home_override=home_dir, respect_env_override=False)
    parent = parent_mgr.create_session()
    parent_mgr.bind_terminal("t1", core_id="core-1")

    # Pointer is published automatically by bind_terminal
    pointer = read_terminal_pointer("core-1", "t1")
    assert pointer is not None
    assert pointer.session_id == parent.info.name
    assert pointer.home_dir == home_dir.resolve()

    # Child process opens a separate SessionManager against the parent's home_dir
    child_mgr = SessionManager(cwd=tmp_path, home_override=home_dir, respect_env_override=False)
    child_link = SessionChildLinkSnapshot(
        parent_session_id=parent.info.name,
        parent_agent_name="main",
        tool_name="hc",
        hc_terminal_id="t2",
        hc_name="code-reviewer",
    )
    child = child_mgr.create_hc_child_session(
        parent.directory,
        parent.info.name,
        child_link,
        alias_slug="code_reviewer",
        label="code-reviewer",
    )
    child_mgr.bind_terminal("t2", core_id="core-1")
    assert child_mgr.current_session is child

    # Child does NOT clutter top-level sessions
    top_level_ids = [info.name for info in parent_mgr.list_sessions()]
    assert top_level_ids == [parent.info.name]
    assert child.directory.parent == parent.directory / "children"

    # Save a turn in the child
    worker_agent = _StubAgent(
        "worker", model="gpt-6.2-astra", input_tokens=200, output_tokens=50, cost_usd=0.02
    )
    await child.save_history(cast("AgentProtocol", worker_agent))

    # Lock-free parent-side inspection succeeds even while child holds its checkpoint lock
    with child._checkpoint_lock():
        listed = parent_mgr.list_child_sessions(parent)
        assert len(listed) == 1
        run = subagent_run_from_session(parent, listed[0])
        assert run is not None
        assert run.tool_name == "hc"
        assert run.hc_terminal_id == "t2"
        assert run.hc_name == "code-reviewer"
        assert run.alias == "01_code_reviewer"
        assert run.status == "running"

    # Parent switches to a new session (releasing its own owner lease on parent)
    other = parent_mgr.create_session()
    assert other.info.name != parent.info.name

    # Deleting the old parent MUST be refused as busy while the hc child is still owned!
    delete_busy = parent_mgr.delete_session_result(parent.info.name)
    assert delete_busy.status == "busy"
    assert child.directory.is_dir()

    # If child closes cleanly, status transitions to completed and parent becomes deletable
    child_mgr.close_terminal_binding_cleanly()
    child_mgr.close()

    run_after_clean = subagent_run_from_session(parent, parent_mgr.list_child_sessions(parent)[0])
    assert run_after_clean is not None
    assert run_after_clean.status == "completed"

    delete_ok = parent_mgr.delete_session_result(parent.info.name)
    assert delete_ok.deleted is True
    parent_mgr.close()


@pytest.mark.asyncio
async def test_hc_child_crash_shows_interrupted_status(tmp_path: Path) -> None:
    home_dir = tmp_path / ".fast-agent"
    parent_mgr = SessionManager(cwd=tmp_path, home_override=home_dir, respect_env_override=False)
    parent = parent_mgr.create_session()

    child_mgr = SessionManager(cwd=tmp_path, home_override=home_dir, respect_env_override=False)
    child = child_mgr.create_hc_child_session(
        parent.directory,
        parent.info.name,
        SessionChildLinkSnapshot(
            parent_session_id=parent.info.name,
            parent_agent_name="main",
            tool_name="hc",
            hc_terminal_id="t9",
            hc_name="crashed-worker",
        ),
        alias_slug="crashed_worker",
    )
    # Mid-turn checkpoint followed by crash (lease dropped before turn finishes) -> interrupted.
    await child.save_history(cast("AgentProtocol", _StubAgent("worker")), checkpoint=True)
    child_mgr.close()

    run = subagent_run_from_session(parent, parent_mgr.list_child_sessions(parent)[0])
    assert run is not None
    assert run.status == "interrupted"

    # Finished turn followed by hc kill while idle at the prompt -> running while live, completed once killed.
    child2_mgr = SessionManager(cwd=tmp_path, home_override=home_dir, respect_env_override=False)
    child2 = child2_mgr.create_hc_child_session(
        parent.directory,
        parent.info.name,
        SessionChildLinkSnapshot(
            parent_session_id=parent.info.name,
            parent_agent_name="main",
            tool_name="hc",
            hc_terminal_id="t10",
            hc_name="finished-worker",
        ),
        alias_slug="finished_worker",
    )
    await child2.save_history(cast("AgentProtocol", _StubAgent("worker2")))
    live_run = subagent_run_from_session(parent, child2)
    assert live_run is not None
    assert live_run.status == "running"

    child2_mgr.close()
    killed_run = subagent_run_from_session(parent, child2)
    assert killed_run is not None
    assert killed_run.status == "completed"
    parent_mgr.close()


def test_resolve_hc_parent_session_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    home_dir = tmp_path / ".fast-agent"

    parent_mgr = SessionManager(cwd=tmp_path, home_override=home_dir, respect_env_override=False)
    parent = parent_mgr.create_session()
    parent_mgr.bind_terminal("t1", core_id="core-xyz")

    request = AgentRunRequest(
        name="worker",
        instruction=None,
        config_path=None,
        server_list=None,
        agent_cards=None,
        card_tools=None,
        model=None,
        message=None,
        prompt_file=None,
        result_file=None,
        resume=None,
        startup_mcp_servers=None,
        mcp_startup_notices=(),
        agent_name="worker",
        target_agent_name=None,
        skills_directory=None,
        home=home_dir,
        no_home=False,
        workspace=tmp_path,
        shell_runtime=False,
        no_shell=False,
        mode="interactive",
        transport="http",
        host="127.0.0.1",
        port=8000,
        tool_description=None,
        tool_name_template=None,
        instance_scope="shared",
        permissions_enabled=True,
        reload=False,
        watch=False,
    )

    monkeypatch.setattr(
        "fast_agent.cli.runtime.hc_recovery.read_hc_self_info",
        lambda: HcSelfInfo(terminal_id="t2", parent="t1", name="worker"),
    )
    monkeypatch.setattr(
        "fast_agent.cli.runtime.hc_recovery.read_hc_core_id",
        lambda: "core-xyz",
    )

    target = resolve_hc_parent_session(request)
    assert target is not None
    assert target.parent_session_id == parent.info.name
    assert target.hc_terminal_id == "t2"
    assert target.hc_name == "worker"

    # Remote parent ("desktop/t1") falls back to None
    monkeypatch.setattr(
        "fast_agent.cli.runtime.hc_recovery.read_hc_self_info",
        lambda: HcSelfInfo(terminal_id="t2", parent="desktop/t1", name="worker"),
    )
    assert resolve_hc_parent_session(request) is None

    # Stale pointer (parent released its lease) falls back to None
    monkeypatch.setattr(
        "fast_agent.cli.runtime.hc_recovery.read_hc_self_info",
        lambda: HcSelfInfo(terminal_id="t2", parent="t1", name="worker"),
    )
    parent_mgr.close()
    assert resolve_hc_parent_session(request) is None


@pytest.mark.asyncio
async def test_hc_usage_and_cost_rollup_excludes_in_process_subagents_and_sums_multi_agent_workers(
    tmp_path: Path,
) -> None:
    home_dir = tmp_path / ".fast-agent"
    parent_mgr = SessionManager(cwd=tmp_path, home_override=home_dir, respect_env_override=False)
    parent = parent_mgr.create_session()

    # In-process subagent child (MUST NOT be double-counted by hc rollup!)
    in_proc = parent_mgr.create_child_session(
        parent,
        SessionChildLinkSnapshot(
            parent_session_id=parent.info.name,
            parent_agent_name="main",
            tool_name="subagent",
        ),
        alias_slug="local_sub",
    )
    await in_proc.save_history(
        cast("AgentProtocol", _StubAgent("sub", input_tokens=999, output_tokens=999, cost_usd=9.99))
    )

    # Multi-agent hc child worker
    child_mgr = SessionManager(cwd=tmp_path, home_override=home_dir, respect_env_override=False)
    hc_child = child_mgr.create_hc_child_session(
        parent.directory,
        parent.info.name,
        SessionChildLinkSnapshot(
            parent_session_id=parent.info.name,
            parent_agent_name="main",
            tool_name="hc",
            hc_terminal_id="t2",
            hc_name="worker-a",
        ),
        alias_slug="worker_a",
    )
    a1 = _StubAgent(
        "router",
        model="model-a",
        input_tokens=100,
        output_tokens=25,
        cache_read_tokens=40,
        tool_calls=1,
        cost_usd=0.01,
    )
    a2 = _StubAgent(
        "specialist",
        model="model-b",
        input_tokens=200,
        output_tokens=75,
        cache_read_tokens=60,
        tool_calls=2,
        cost_usd=0.03,
    )
    await hc_child.save_history(
        cast("AgentProtocol", a1),
        agent_registry={
            "router": cast("AgentProtocol", a1),
            "specialist": cast("AgentProtocol", a2),
        },
    )
    child_mgr.close()

    rollups = collect_hc_child_usage_rollups(parent_mgr)
    assert len(rollups) == 1
    rollup = rollups[0]
    assert rollup.parent_agent_name == "main"
    assert rollup.child_count == 1
    assert rollup.input_tokens == 300
    assert rollup.output_tokens == 100
    assert rollup.cache_read_tokens == 100
    assert rollup.tool_calls == 3
    assert rollup.model == "multiple"
    assert rollup.cost_usd == pytest.approx(0.04)

    # Render combined markdown table with parent + hc child + cost column
    parent_agent = _StubAgent(
        "main",
        model="model-main",
        input_tokens=500,
        output_tokens=100,
        cache_read_tokens=250,
        tool_calls=2,
        cost_usd=0.06,
    )
    markdown = format_usage_markdown(
        {"main": parent_agent},
        extra_rows=usage_rows_from_hc_rollups(rollups),
    )
    assert "| Cost |" in markdown
    assert "main › hc" in markdown
    assert "$0.04" in markdown
    assert "$0.10" in markdown  # TOTAL = $0.06 + $0.04
    assert "999" not in markdown  # in-process subagent NOT double-counted
    parent_mgr.close()


def test_v7_snapshot_migrates_to_v8() -> None:
    payload = {
        "schema_version": 7,
        "session_id": "2604141705-AbCd12",
        "created_at": "2026-04-14T17:05:00",
        "last_activity": "2026-04-14T17:09:00",
        "continuation": {"active_agent": "main", "agents": {}},
    }
    snapshot = load_session_snapshot(payload)
    assert snapshot.schema_version == SESSION_SNAPSHOT_SCHEMA_VERSION == 8


@pytest.mark.asyncio
async def test_hc_child_creation_does_not_contend_with_parent_checkpoint_and_rechecks_liveness(
    tmp_path: Path,
) -> None:
    home_dir = tmp_path / ".fast-agent"
    parent_mgr = SessionManager(cwd=tmp_path, home_override=home_dir, respect_env_override=False)
    parent = parent_mgr.create_session()
    parent_mgr.bind_terminal("t1", core_id="core-1")

    child_mgr = SessionManager(cwd=tmp_path, home_override=home_dir, respect_env_override=False)
    # Must succeed even while parent is mid-checkpoint (no parent checkpoint lock taken!)
    with parent._checkpoint_lock():
        child = child_mgr.create_hc_child_session(
            parent.directory,
            parent.info.name,
            SessionChildLinkSnapshot(
                parent_session_id=parent.info.name,
                parent_agent_name="main",
                tool_name="hc",
                hc_terminal_id="t2",
                hc_name="worker",
            ),
            alias_slug="worker",
            expected_parent_terminal_id="t1",
            expected_core_id="core-1",
        )
    assert child.directory.parent == parent.directory / "children"

    # Once parent closes its terminal binding, a late child publication MUST be rejected
    parent_mgr.close_terminal_binding_cleanly()
    parent_mgr.close()
    late_mgr = SessionManager(cwd=tmp_path, home_override=home_dir, respect_env_override=False)
    with pytest.raises(ValueError, match="no longer"):
        late_mgr.create_hc_child_session(
            parent.directory,
            parent.info.name,
            SessionChildLinkSnapshot(
                parent_session_id=parent.info.name,
                parent_agent_name="main",
                tool_name="hc",
                hc_terminal_id="t3",
                hc_name="late",
            ),
            alias_slug="late",
            expected_parent_terminal_id="t1",
            expected_core_id="core-1",
        )
    child_mgr.close()
    late_mgr.close()


@pytest.mark.asyncio
async def test_hc_child_fork_is_resumable_and_displaced_child_is_finalized(tmp_path: Path) -> None:
    home_dir = tmp_path / ".fast-agent"
    parent_mgr = SessionManager(cwd=tmp_path, home_override=home_dir, respect_env_override=False)
    parent = parent_mgr.create_session()

    child_mgr = SessionManager(cwd=tmp_path, home_override=home_dir, respect_env_override=False)
    child = child_mgr.create_hc_child_session(
        parent.directory,
        parent.info.name,
        SessionChildLinkSnapshot(
            parent_session_id=parent.info.name,
            parent_agent_name="main",
            tool_name="hc",
            hc_terminal_id="t2",
            hc_name="forker",
        ),
        alias_slug="forker",
    )
    await child.save_history(cast("AgentProtocol", _StubAgent("worker")))

    forked = child_mgr.fork_current_session()
    assert forked is not None
    forked_snapshot = forked.load_snapshot_unlocked()
    assert forked_snapshot.execution.resumable is True
    assert forked_snapshot.execution.child_link is None

    # Displaced child was finalized to completed (not left as interrupted)
    run = subagent_run_from_session(parent, parent_mgr.list_child_sessions(parent)[0])
    assert run is not None
    assert run.status == "completed"

    child_mgr.close()
    parent_mgr.close()


@pytest.mark.asyncio
async def test_hc_usage_rollup_marks_partial_sibling_dimensions_incomplete(tmp_path: Path) -> None:
    home_dir = tmp_path / ".fast-agent"
    parent_mgr = SessionManager(cwd=tmp_path, home_override=home_dir, respect_env_override=False)
    parent = parent_mgr.create_session()

    # Sibling 1: fully measured with cost
    c1_mgr = SessionManager(cwd=tmp_path, home_override=home_dir, respect_env_override=False)
    c1 = c1_mgr.create_hc_child_session(
        parent.directory,
        parent.info.name,
        SessionChildLinkSnapshot(
            parent_session_id=parent.info.name,
            parent_agent_name="main",
            tool_name="hc",
            hc_terminal_id="t2",
            hc_name="measured",
        ),
        alias_slug="measured",
    )
    await c1.save_history(
        cast(
            "AgentProtocol",
            _StubAgent("w1", input_tokens=100, output_tokens=20, tool_calls=1, cost_usd=0.05),
        )
    )
    c1_mgr.close()

    # Sibling 2: partial usage (no prompt_tokens, no cost_usd, but 2 tool calls)
    c2_mgr = SessionManager(cwd=tmp_path, home_override=home_dir, respect_env_override=False)
    c2 = c2_mgr.create_hc_child_session(
        parent.directory,
        parent.info.name,
        SessionChildLinkSnapshot(
            parent_session_id=parent.info.name,
            parent_agent_name="main",
            tool_name="hc",
            hc_terminal_id="t3",
            hc_name="partial",
        ),
        alias_slug="partial",
    )
    partial_agent = _StubAgent("w2", input_tokens=0, output_tokens=0, cost_usd=None)
    partial_agent.usage_accumulator.add_turn(
        TurnUsage(
            provider=Provider.RESPONSES,
            usage_schema=UsageSchema.OPENAI_RESPONSES,
            model="partial-model",
            prompt=PromptTokenUsage(total=None),
            completion=CompletionTokenUsage(total=30),
            tool_calls=2,
            cost_usd=None,
        )
    )
    await c2.save_history(cast("AgentProtocol", partial_agent))
    c2_mgr.close()

    rollups = collect_hc_child_usage_rollups(parent_mgr)
    assert len(rollups) == 1
    rollup = rollups[0]
    assert rollup.child_count == 2
    assert rollup.input_tokens is None
    assert rollup.output_tokens == 50
    assert rollup.tool_calls == 3
    assert rollup.cost_usd is None
    parent_mgr.close()


@pytest.mark.asyncio
async def test_collect_hc_child_user_turn_usage_reads_persisted_attempts_and_survives_resume(
    tmp_path: Path,
) -> None:
    import json

    from fast_agent.constants import FAST_AGENT_SYNTHETIC_FINAL_CHANNEL, FAST_AGENT_USAGE
    from fast_agent.llm.usage_tracking import UsageReport, UserTurnUsage
    from fast_agent.session.subagent_runs import (
        collect_hc_child_user_turn_usage,
        with_hc_child_user_turn_usage,
    )

    parent_mgr = SessionManager(cwd=tmp_path)
    parent = parent_mgr.create_session(name="parent")
    parent_mgr.bind_terminal("t1", core_id="c1")

    def _usage_payload(model: str, prompt: int, output: int) -> str:
        report = UsageReport(
            provider_attempts=[
                TurnUsage(
                    provider=Provider.RESPONSES,
                    usage_schema=UsageSchema.OPENAI_RESPONSES,
                    model=model,
                    prompt=PromptTokenUsage(total=prompt, cache_read=prompt // 2),
                    completion=CompletionTokenUsage(total=output),
                    tool_calls=1,
                )
            ]
        )
        return report.model_dump_json(by_alias=True)

    def _write_history(child_dir: Path, filename: str, messages: list[dict[str, object]]) -> None:
        (child_dir / filename).write_text(
            json.dumps({"messages": messages}),
            encoding="utf-8",
        )

    # In-process subagent child — must be excluded from hc child ledgers.
    sub_child = parent_mgr.create_child_session(
        parent,
        SessionChildLinkSnapshot(
            parent_session_id=parent.info.name,
            parent_agent_name="main",
            tool_name="subagent",
        ),
        alias_slug="inproc",
    )
    await sub_child.save_history(
        cast("AgentProtocol", _StubAgent("sub", input_tokens=999, output_tokens=99))
    )
    _write_history(
        sub_child.directory,
        "history_sub.json",
        [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
                "channels": {
                    FAST_AGENT_USAGE: [
                        {"type": "text", "text": _usage_payload("inproc-model", 999, 99)}
                    ]
                },
            }
        ],
    )

    # First hc child (with a real attempt, a synthetic final to skip, and an undeclared previous file to ignore).
    c1_mgr = SessionManager(cwd=tmp_path)
    c1 = c1_mgr.create_hc_child_session(
        parent.directory,
        parent.info.name,
        SessionChildLinkSnapshot(
            parent_session_id=parent.info.name,
            parent_agent_name="main",
            tool_name="hc",
            hc_terminal_id="t2",
            hc_name="worker-1",
        ),
        alias_slug="worker-1",
    )
    await c1.save_history(
        cast("AgentProtocol", _StubAgent("w1", input_tokens=100, output_tokens=20))
    )
    _write_history(
        c1.directory,
        "history_w1.json",
        [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "step 1"}],
                "channels": {
                    FAST_AGENT_USAGE: [{"type": "text", "text": _usage_payload("model-a", 100, 20)}]
                },
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "synthetic"}],
                "channels": {
                    FAST_AGENT_SYNTHETIC_FINAL_CHANNEL: [
                        {"type": "text", "text": "tool_result_passthrough"}
                    ],
                    FAST_AGENT_USAGE: [
                        {"type": "text", "text": _usage_payload("model-a", 500, 50)}
                    ],
                },
            },
        ],
    )
    _write_history(
        c1.directory,
        "history_w1_previous.json",
        [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "old"}],
                "channels": {
                    FAST_AGENT_USAGE: [{"type": "text", "text": _usage_payload("model-a", 777, 77)}]
                },
            }
        ],
    )
    c1_mgr.close()

    # Second hc child.
    c2_mgr = SessionManager(cwd=tmp_path)
    c2 = c2_mgr.create_hc_child_session(
        parent.directory,
        parent.info.name,
        SessionChildLinkSnapshot(
            parent_session_id=parent.info.name,
            parent_agent_name="main",
            tool_name="hc",
            hc_terminal_id="t3",
            hc_name="worker-2",
        ),
        alias_slug="worker-2",
    )
    await c2.save_history(
        cast("AgentProtocol", _StubAgent("w2", input_tokens=200, output_tokens=40))
    )
    _write_history(
        c2.directory,
        "history_w2.json",
        [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "step 1"}],
                "channels": {
                    FAST_AGENT_USAGE: [{"type": "text", "text": _usage_payload("model-b", 200, 40)}]
                },
            }
        ],
    )
    c2_mgr.close()
    parent_mgr.close()

    # Simulate resuming the parent session in a brand-new SessionManager.
    resumed_mgr = SessionManager(cwd=tmp_path)
    resumed_parent = resumed_mgr.get_session(parent.info.name)
    assert resumed_parent is not None

    hc_turn = collect_hc_child_user_turn_usage(resumed_mgr, parent=resumed_parent)
    assert hc_turn is not None
    assert hc_turn.agent_name == "hc"
    assert [ledger.label for ledger in hc_turn.ledgers] == ["hc:worker-1", "hc:worker-2"]
    assert [attempt.model for attempt in hc_turn.attempts] == ["model-a", "model-b"]
    assert sum(attempt.prompt.total or 0 for attempt in hc_turn.attempts) == 300
    assert sum(attempt.completion.total or 0 for attempt in hc_turn.attempts) == 60

    parent_turn = UserTurnUsage(agent_name="main", attempts=())
    combined = with_hc_child_user_turn_usage((parent_turn,), resumed_mgr, parent=resumed_parent)
    assert len(combined) == 2
    assert combined[0] is parent_turn
    assert combined[1] == hc_turn
    resumed_mgr.close()
