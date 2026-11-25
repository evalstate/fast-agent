"""Pre-prompt herdr-core (``hc``) terminal recovery coordinator for interactive CLI runs.

Runs before ``OSC 7501`` is activated so ``herdr-core`` does not flush queued
``stopped_by_restart`` notices into a fresh session before the user chooses
whether to resume the interrupted session bound to this terminal or start fresh.
"""

from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from rich.markup import escape

from fast_agent.core.exceptions import PromptExitError
from fast_agent.integrations.hc_children import (
    HcTerminalChildren,
    hc_environment,
    read_hc_core_id,
    read_hc_self_info,
    read_hc_terminal_children,
    remove_hc_terminals,
)
from fast_agent.paths import resolve_home_paths
from fast_agent.session.locking import SessionLockStore
from fast_agent.session.session_manager import (
    SessionInfo,
    SessionManager,
    display_session_name,
    session_metadata_title,
)
from fast_agent.session.snapshot import load_session_snapshot
from fast_agent.session.terminal_pointers import HcParentSessionTarget, read_terminal_pointer
from fast_agent.ui.console import console

if TYPE_CHECKING:
    from collections.abc import Callable

    from fast_agent.cli.runtime.run_request import AgentRunRequest

type HcRecoveryAction = Literal["resume", "fresh_clear", "fresh", "noop"]

_MAX_LISTED_CHILDREN = 6


@dataclass(frozen=True, slots=True)
class HcRecoveryPlan:
    """Observed interrupted-session and hc-child state for the current terminal."""

    terminal_id: str
    core_id: str | None
    interrupted_session: SessionInfo | None
    children: HcTerminalChildren | None
    interrupted_updated_at: datetime | None = None
    explicit_resume_target: str | None = None

    @property
    def stopped_by_restart(self) -> tuple[str, ...]:
        if self.children is None:
            return ()
        return tuple(child.address for child in self.children.stopped_by_restart)

    @property
    def needs_prompt(self) -> bool:
        return self.interrupted_session is not None or bool(self.stopped_by_restart)


@dataclass(frozen=True, slots=True)
class HcRecoveryOutcome:
    """Result of evaluating and applying startup recovery."""

    action: HcRecoveryAction = "noop"
    resumed_session_id: str | None = None
    removed_terminals: tuple[str, ...] = ()
    failed_removals: tuple[str, ...] = ()


def should_check_hc_recovery(
    request: AgentRunRequest,
    *,
    stdin_is_tty: bool | None = None,
    stdout_is_tty: bool | None = None,
) -> bool:
    """Return True when an interactive CLI launch should check for hc terminal recovery."""
    if request.mode != "interactive" or not request.is_repl or request.no_home:
        return False
    in_tty = sys.stdin.isatty() if stdin_is_tty is None else stdin_is_tty
    out_tty = sys.stdout.isatty() if stdout_is_tty is None else stdout_is_tty
    if not (in_tty and out_tty):
        return False
    return hc_environment() is not None


def inspect_hc_recovery(request: AgentRunRequest) -> HcRecoveryPlan | None:
    """Passively inspect the current hc terminal and local session store.

    Blocking; call off the event loop. Never calls ``hc get`` or ``hc wait`` in
    parent context so pending parent notices are not acknowledged.
    """
    env = hc_environment()
    if env is None:
        return None
    core_id = read_hc_core_id()
    children = read_hc_terminal_children(env.terminal_id)
    interrupted: SessionInfo | None = None
    interrupted_updated_at: datetime | None = None
    explicit_resume_target: str | None = None

    if request.resume is not None:
        if children is None or not children.stopped_by_restart:
            return HcRecoveryPlan(
                terminal_id=env.terminal_id,
                core_id=core_id,
                interrupted_session=None,
                children=children,
            )
        manager = SessionManager(cwd=request.workspace, home_override=request.home)
        interrupted = manager.find_interrupted_terminal_session(env.terminal_id, core_id=core_id)
        if request.resume in ("", "latest"):
            sessions = manager.list_sessions()
            target_id = sessions[0].name if sessions else None
        else:
            target_id = manager.resolve_session_name(request.resume)
        if interrupted is not None and target_id == interrupted.name:
            return HcRecoveryPlan(
                terminal_id=env.terminal_id,
                core_id=core_id,
                interrupted_session=None,
                children=None,
            )
        explicit_resume_target = (
            display_session_name(target_id) if target_id else request.resume or "latest"
        )
        return HcRecoveryPlan(
            terminal_id=env.terminal_id,
            core_id=core_id,
            interrupted_session=None,
            children=children,
            explicit_resume_target=explicit_resume_target,
        )

    if not request.force_model_picker and request.model_base_url is None:
        manager = SessionManager(cwd=request.workspace, home_override=request.home)
        interrupted = manager.find_interrupted_terminal_session(env.terminal_id, core_id=core_id)
        if interrupted is not None:
            session = manager.get_session(interrupted.name)
            if session is not None:
                try:
                    attachment = session.load_snapshot().execution.terminal_attachment
                except Exception:
                    attachment = None
                if attachment is not None:
                    interrupted_updated_at = attachment.updated_at
    return HcRecoveryPlan(
        terminal_id=env.terminal_id,
        core_id=core_id,
        interrupted_session=interrupted,
        children=children,
        interrupted_updated_at=interrupted_updated_at,
    )


def _format_age(timestamp: datetime) -> str:
    now = datetime.now(timestamp.tzinfo)
    seconds = max(0, int((now - timestamp).total_seconds()))
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h ago"
    return f"{hours // 24}d ago"


def _format_child_summary(plan: HcRecoveryPlan) -> str:
    assert plan.children is not None
    stopped = plan.children.stopped_by_restart
    shown = ", ".join(child.label for child in stopped[:_MAX_LISTED_CHILDREN])
    remaining = len(stopped) - _MAX_LISTED_CHILDREN
    if remaining > 0:
        return f"{shown}, +{remaining} more"
    return shown


def _available_choices(plan: HcRecoveryPlan) -> tuple[dict[str, HcRecoveryAction], str]:
    choices: dict[str, HcRecoveryAction] = {}
    if plan.interrupted_session is not None:
        choices["r"] = "resume"
    if plan.stopped_by_restart:
        choices["c"] = "fresh_clear"
    choices["n"] = "fresh"
    default_key = "r" if "r" in choices else "c" if "c" in choices else "n"
    return choices, default_key


def _render_recovery_summary(plan: HcRecoveryPlan) -> None:
    parts: list[str] = []
    if plan.interrupted_session is not None:
        parts.append("an interrupted session")
    stopped_count = len(plan.stopped_by_restart)
    if stopped_count == 1:
        parts.append("1 stopped_by_restart hc child")
    elif stopped_count > 1:
        parts.append(f"{stopped_count} stopped_by_restart hc children")

    header = " and ".join(parts)
    console.print(f"[yellow]Terminal {escape(plan.terminal_id)} has {header}:[/yellow]")

    if plan.interrupted_session is not None:
        info = plan.interrupted_session
        short_id = display_session_name(info.name)
        title = session_metadata_title(info.metadata)
        preview = f' — "{title[:72]}"' if title else ""
        age = _format_age(info.last_activity)
        console.print(
            f"  [bold]Session:[/bold] [cyan]{escape(short_id)}[/cyan]"
            f"{escape(preview)} [dim]({age})[/dim]"
        )

    if plan.children is not None and plan.children.stopped_by_restart:
        console.print(
            f"  [bold]Stopped:[/bold] {escape(_format_child_summary(plan))} "
            "[dim](retained; notices may arrive when idle)[/dim]"
        )

    if plan.children is not None and plan.children.live:
        live_labels = ", ".join(child.label for child in plan.children.live[:_MAX_LISTED_CHILDREN])
        console.print(f"  [dim]Live children (untouched): {escape(live_labels)}[/dim]")


def _prompt_recovery_choice(plan: HcRecoveryPlan) -> HcRecoveryAction:
    choices, default_key = _available_choices(plan)
    _render_recovery_summary(plan)
    console.print()
    if plan.interrupted_session is not None:
        short_id = display_session_name(plan.interrupted_session.name)
        suffix = " [dim](default)[/dim]" if default_key == "r" else ""
        console.print(
            rf"  [bold]\[r][/bold] Resume [cyan]{escape(short_id)}[/cyan] "
            f"(deliver pending hc notices there){suffix}"
        )
    if plan.explicit_resume_target is not None:
        target_label = escape(plan.explicit_resume_target)
        if plan.stopped_by_restart:
            suffix = " [dim](default)[/dim]" if default_key == "c" else ""
            console.print(
                rf"  [bold]\[c][/bold] Clear stopped_by_restart children (hc rm) & resume [cyan]{target_label}[/cyan]{suffix}"
            )
        console.print(
            rf"  [bold]\[n][/bold] Resume [cyan]{target_label}[/cyan] (leave hc children as-is)"
        )
    else:
        if plan.stopped_by_restart:
            suffix = " [dim](default)[/dim]" if default_key == "c" else ""
            console.print(
                rf"  [bold]\[c][/bold] Start fresh & clear stopped_by_restart children (hc rm){suffix}"
            )
        console.print(r"  [bold]\[n][/bold] Start fresh (leave hc children as-is)")

    keys_label = "/".join(key.upper() if key == default_key else key for key in choices)
    while True:
        try:
            console.print(rf"Choice \[{keys_label}]: ", end="")
            raw = sys.stdin.readline()
        except KeyboardInterrupt as exc:
            raise PromptExitError("Cancelled startup recovery") from exc
        if raw == "":
            raise PromptExitError("Cancelled startup recovery")
        selected = raw.strip().lower() or default_key
        if selected in choices:
            return choices[selected]
        console.print(f"[dim]Please enter one of: {', '.join(choices)}[/dim]")


async def resolve_hc_startup_recovery(
    request: AgentRunRequest,
    *,
    prompt_choice: Callable[[HcRecoveryPlan], HcRecoveryAction] | None = None,
) -> HcRecoveryOutcome:
    """Inspect and resolve interrupted hc terminal state before OSC 7501 activates."""
    if not should_check_hc_recovery(request):
        return HcRecoveryOutcome()

    plan = await asyncio.to_thread(inspect_hc_recovery, request)
    if plan is None or not plan.needs_prompt:
        return HcRecoveryOutcome()

    chooser = prompt_choice or _prompt_recovery_choice
    action = chooser(plan)

    if action == "resume" and plan.interrupted_session is not None:
        request.resume = plan.interrupted_session.name
        return HcRecoveryOutcome(
            action="resume",
            resumed_session_id=plan.interrupted_session.name,
        )

    if plan.interrupted_session is not None:
        session_id = plan.interrupted_session.name
        await asyncio.to_thread(
            _retire_declined_session,
            request,
            session_id,
            plan.terminal_id,
            plan.core_id,
            plan.interrupted_updated_at,
        )

    if action == "fresh_clear":
        removed, failed = await asyncio.to_thread(remove_hc_terminals, plan.stopped_by_restart)
        if failed:
            console.print(
                "[yellow]Warning: could not remove hc terminal(s): "
                f"{escape(', '.join(failed))}[/yellow]"
            )
        return HcRecoveryOutcome(
            action="fresh_clear",
            removed_terminals=tuple(removed),
            failed_removals=tuple(failed),
        )

    return HcRecoveryOutcome(action="fresh")


def _retire_declined_session(
    request: AgentRunRequest,
    session_id: str,
    terminal_id: str,
    core_id: str | None,
    expected_updated_at: datetime | None = None,
) -> None:
    manager = SessionManager(
        cwd=request.workspace,
        home_override=request.home,
    )
    manager.retire_terminal_binding(
        session_id,
        terminal_id,
        core_id=core_id,
        expected_updated_at=expected_updated_at,
    )


def resolve_hc_parent_session(request: AgentRunRequest) -> HcParentSessionTarget | None:
    """Resolve a live local parent ``fast-agent`` session when launched via ``hc run``.

    Returns ``None`` (falling back to a normal top-level session) whenever this
    process is not a fresh interactive ``hc`` child, its parent is on another
    machine, its parent is not a live ``fast-agent`` session, or the pointer
    cannot be verified against the parent session's owner lease and attachment.
    """
    if (
        request.mode != "interactive"
        or not request.is_repl
        or request.resume is not None
        or request.no_home
    ):
        return None

    self_info = read_hc_self_info()
    if self_info is None or self_info.parent is None or "/" in self_info.parent:
        return None

    core_id = read_hc_core_id()
    if not core_id:
        return None

    pointer = read_terminal_pointer(core_id, self_info.parent)
    if pointer is None:
        return None

    metadata_file = pointer.session_dir / "session.json"
    if not metadata_file.is_file():
        return None

    try:
        snapshot = load_session_snapshot(json.loads(metadata_file.read_text(encoding="utf-8")))
    except Exception:
        return None

    if snapshot.session_id != pointer.session_id:
        return None

    attachment = snapshot.execution.terminal_attachment
    if (
        attachment is None
        or not attachment.active
        or attachment.terminal_id != self_info.parent
        or attachment.core_id != core_id
    ):
        return None

    sessions_dir = resolve_home_paths(override=pointer.home_dir).sessions
    if not SessionLockStore(sessions_dir).is_owned(pointer.session_id):
        return None

    parent_agent = snapshot.continuation.active_agent or pointer.agent_name or "default"
    return HcParentSessionTarget(
        home_dir=pointer.home_dir,
        parent_session_id=pointer.session_id,
        parent_session_dir=pointer.session_dir,
        parent_agent_name=parent_agent,
        hc_terminal_id=self_info.terminal_id,
        hc_name=self_info.name,
        parent_terminal_id=self_info.parent,
        core_id=core_id,
    )
