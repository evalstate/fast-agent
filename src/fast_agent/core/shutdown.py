"""Shared cooperative shutdown budget and timing diagnostics."""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fast_agent.core.logging.logger import get_logger
from fast_agent.core.runtime_diagnostics import write_runtime_trace

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = get_logger(__name__)

TUI_SHUTDOWN_TIMEOUT_SECONDS = 2.0
_SHUTDOWN_TRACE_ENV_VAR = "FAST_AGENT_SHUTDOWN_DEBUG_TRACE"
_INTERACTIVE_TRACE_ENV_VAR = "FAST_AGENT_INTERACTIVE_DEBUG_TRACE"


def write_shutdown_trace(event: str, **fields: object) -> None:
    trace_path = os.getenv(_SHUTDOWN_TRACE_ENV_VAR, "").strip()
    if not trace_path:
        trace_path = os.getenv(_INTERACTIVE_TRACE_ENV_VAR, "").strip()
    write_runtime_trace(trace_path, event, **fields)


@dataclass(slots=True)
class ShutdownBudget:
    """One monotonic deadline shared by every cleanup phase."""

    timeout_seconds: float
    reason: str
    _started_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("shutdown timeout must be greater than zero")
        write_shutdown_trace(
            "shutdown.begin",
            reason=self.reason,
            budget_ms=round(self.timeout_seconds * 1000, 3),
        )

    @property
    def deadline(self) -> float:
        return self._started_at + self.timeout_seconds

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    async def run(self, phase: str, operation: Callable[[], Awaitable[None]]) -> bool:
        """Run one phase within the remaining budget and report whether it completed."""
        phase_started_at = time.monotonic()
        remaining = self.remaining_seconds()
        write_shutdown_trace(
            "shutdown.phase.start",
            phase=phase,
            remaining_ms=round(remaining * 1000, 3),
        )
        if remaining <= 0:
            self._trace_phase_end(phase, phase_started_at, outcome="skipped")
            return False

        outcome = "completed"
        try:
            async with asyncio.timeout_at(self.deadline):
                await operation()
        except TimeoutError:
            outcome = "timed_out"
            logger.warning(
                "Shutdown deadline reached",
                phase=phase,
                timeout_seconds=self.timeout_seconds,
            )
        except BaseException:
            outcome = "failed"
            raise
        finally:
            if outcome == "completed" and self.remaining_seconds() <= 0:
                outcome = "deadline_exhausted"
            self._trace_phase_end(phase, phase_started_at, outcome=outcome)
        return outcome == "completed"

    def complete(self) -> None:
        elapsed = time.monotonic() - self._started_at
        write_shutdown_trace(
            "shutdown.complete",
            reason=self.reason,
            elapsed_ms=round(elapsed * 1000, 3),
            budget_ms=round(self.timeout_seconds * 1000, 3),
            budget_exhausted=self.remaining_seconds() <= 0,
        )

    def _trace_phase_end(self, phase: str, started_at: float, *, outcome: str) -> None:
        write_shutdown_trace(
            "shutdown.phase.end",
            phase=phase,
            outcome=outcome,
            elapsed_ms=round((time.monotonic() - started_at) * 1000, 3),
            remaining_ms=round(self.remaining_seconds() * 1000, 3),
        )
