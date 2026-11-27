"""Standalone Streamable HTTP MCP Events (SEP-3415) playground server.

Provides a poll-based SEP-3415 MCP server with isolated per-room endpoints at
``/r/<room_id>/mcp`` (plus a ``/mcp`` public fallback) alongside a browser
control plane for triggering events, testing truncated gaps, subscription
revocations, ``maxAgeMs``, ``maxEvents``, and ``--resume``.

Run locally:
    uv run examples/mcp/events/playground/app.py --port 8765

Deploy to Hugging Face Spaces (Docker Space, port 7860):
    Upload the contents of this directory to a Docker Space repository.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import contextvars
import json
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import uvicorn
from mcp.server.extension import Extension, MethodBinding
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.shared.exceptions import MCPError
from mcp_types import (
    INVALID_PARAMS,
    PaginatedRequestParams,
    PaginatedResult,
    RequestParams,
    Result,
)
from pydantic import BaseModel, ConfigDict, Field
from starlette.applications import Starlette
from starlette.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Route

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from starlette.requests import Request
    from starlette.types import ASGIApp, Receive, Scope, Send

# ---------------------------------------------------------------------------
# SEP-3415 wire models (inlined so this server has zero fast-agent dependency)
# ---------------------------------------------------------------------------

EVENTS_EXTENSION = "io.modelcontextprotocol/events"
EVENTS_NOT_FOUND = -32023
EVENTS_FORBIDDEN = -32024


class EventDescriptor(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    description: str | None = None
    delivery: list[str]
    input_schema: dict[str, Any] | None = Field(default=None, alias="inputSchema")
    payload_schema: dict[str, Any] | None = Field(default=None, alias="payloadSchema")
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")


class EventOccurrence(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    event_id: str = Field(alias="eventId")
    name: str
    timestamp: str
    data: dict[str, Any]
    cursor: str | None = None
    meta: dict[str, Any] | None = Field(default=None, alias="_meta")


class ListEventsResult(PaginatedResult):
    events: list[EventDescriptor]
    result_type: Literal["complete"] = Field(default="complete", alias="resultType")


class PollEventsRequestParams(RequestParams):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    cursor: str | None = None
    max_age_ms: int | None = Field(default=None, alias="maxAgeMs")
    max_events: int | None = Field(default=None, alias="maxEvents")


class PollEventsResult(Result):
    events: list[EventOccurrence] = Field(default_factory=list)
    cursor: str | None = None
    truncated: bool = False
    has_more: bool = Field(default=False, alias="hasMore")
    next_poll_ms: int | None = Field(default=None, alias="nextPollMs")
    result_type: Literal["complete"] = Field(default="complete", alias="resultType")


# ---------------------------------------------------------------------------
# Per-room in-memory playground state
# ---------------------------------------------------------------------------

MAX_STREAM_HISTORY = 500
MAX_POLL_LOG = 200
MAX_RECENT_EVENTS = 100
ROOM_TTL_SECONDS = 4 * 3600
MAX_ROOMS = 500
PUBLIC_ROOM_ID = "public"
ROOM_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{2,31}$")
ROOM_MCP_PATH_RE = re.compile(r"^/r/([^/]+)/mcp/?$")

_ROOM_ADJECTIVES = (
    "amber",
    "cobalt",
    "crisp",
    "delta",
    "echo",
    "glass",
    "linen",
    "lunar",
    "matte",
    "prism",
    "quartz",
    "signal",
    "silver",
    "slate",
    "solar",
    "vector",
)
_ROOM_NOUNS = (
    "beacon",
    "bridge",
    "bus",
    "clock",
    "dial",
    "gate",
    "latch",
    "meter",
    "node",
    "pulse",
    "relay",
    "scope",
    "switch",
    "trace",
    "valve",
    "wave",
)


@dataclass(frozen=True)
class StoredEvent:
    event_id: str
    stream_seq: int
    name: str
    filter_key: str
    timestamp: datetime
    data: dict[str, Any]

    def to_occurrence(self) -> EventOccurrence:
        return EventOccurrence(
            event_id=self.event_id,
            name=self.name,
            timestamp=self.timestamp.isoformat(),
            data=self.data,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "name": self.name,
            "filter_key": self.filter_key,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }


@dataclass
class PlaygroundState:
    room_id: str = PUBLIC_ROOM_ID
    poll_interval_ms: int = field(
        default_factory=lambda: max(250, int(os.environ.get("POLL_INTERVAL_MS", "2000")))
    )
    streams: dict[tuple[str, str], list[StoredEvent]] = field(default_factory=dict)
    stream_totals: dict[tuple[str, str], int] = field(default_factory=dict)
    pending_gaps: set[tuple[str, str]] = field(default_factory=set)
    forbidden_keys: set[tuple[str, str]] = field(default_factory=set)
    poll_log: list[dict[str, Any]] = field(default_factory=list)
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    last_polled: dict[tuple[str, str], float] = field(default_factory=dict)
    event_seq: int = 0
    last_touched: float = field(default_factory=time.monotonic)
    subscribers: set[asyncio.Queue[dict[str, Any]]] = field(default_factory=set)

    def touch(self) -> None:
        self.last_touched = time.monotonic()

    def notify(self, payload: dict[str, Any]) -> None:
        for queue in list(self.subscribers):
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(payload)

    def active_streams(self) -> list[str]:
        window = max((self.poll_interval_ms * 4) / 1000.0, 20.0)
        cutoff = time.monotonic() - window
        return [
            f"{k[0]}:{k[1]}"
            for k, ts in sorted(self.last_polled.items(), key=lambda item: item[1], reverse=True)
            if ts >= cutoff
        ]

    def emit(self, event_name: str, filter_key: str, data: dict[str, Any]) -> StoredEvent:
        self.touch()
        key = (event_name, filter_key)
        self.event_seq += 1
        stream_seq = self.stream_totals.get(key, 0) + 1
        self.stream_totals[key] = stream_seq
        record = StoredEvent(
            event_id=f"ev-{self.event_seq}",
            stream_seq=stream_seq,
            name=event_name,
            filter_key=filter_key,
            timestamp=datetime.now(UTC),
            data=data,
        )
        bucket = self.streams.setdefault(key, [])
        bucket.append(record)
        if len(bucket) > MAX_STREAM_HISTORY:
            del bucket[:-MAX_STREAM_HISTORY]

        self.recent_events.append(record.to_dict())
        if len(self.recent_events) > MAX_RECENT_EVENTS:
            del self.recent_events[:-MAX_RECENT_EVENTS]
        self.notify({"kind": "emit", "name": event_name, "filter_key": filter_key})
        return record

    def reset(self) -> None:
        # Keep stream_totals and event_seq monotonic so connected clients see
        # truncated=True and fresh event IDs instead of silently skipping events.
        self.touch()
        self.streams.clear()
        self.pending_gaps.clear()
        self.forbidden_keys.clear()
        self.poll_log.clear()
        self.recent_events.clear()
        self.last_polled.clear()
        self.notify({"kind": "state"})


class RoomRegistry:
    def __init__(self) -> None:
        self._rooms: dict[str, PlaygroundState] = {
            PUBLIC_ROOM_ID: PlaygroundState(room_id=PUBLIC_ROOM_ID)
        }

    def new_room_id(self) -> str:
        for _ in range(16):
            candidate = (
                f"{secrets.choice(_ROOM_ADJECTIVES)}-"
                f"{secrets.choice(_ROOM_NOUNS)}-"
                f"{secrets.randbelow(90) + 10}"
            )
            if candidate not in self._rooms:
                return candidate
        return f"room-{secrets.token_hex(4)}"

    def validate_room_id(self, room_id: str) -> str | None:
        cleaned = room_id.strip().lower()
        if ROOM_ID_RE.match(cleaned):
            return cleaned
        return None

    def get_or_create(self, room_id: str) -> PlaygroundState:
        self._prune()
        room = self._rooms.get(room_id)
        if room is None:
            room = PlaygroundState(room_id=room_id)
            self._rooms[room_id] = room
        else:
            room.touch()
        return room

    def _prune(self) -> None:
        now = time.monotonic()
        expired = [
            rid
            for rid, room in self._rooms.items()
            if rid != PUBLIC_ROOM_ID and (now - room.last_touched) > ROOM_TTL_SECONDS
        ]
        for rid in expired:
            self._rooms.pop(rid, None)

        if len(self._rooms) > MAX_ROOMS:
            evictable = sorted(
                (
                    (room.last_touched, rid)
                    for rid, room in self._rooms.items()
                    if rid != PUBLIC_ROOM_ID
                ),
                key=lambda item: item[0],
            )
            overflow = len(self._rooms) - MAX_ROOMS
            for _, rid in evictable[:overflow]:
                self._rooms.pop(rid, None)


registry = RoomRegistry()
current_room_var: contextvars.ContextVar[PlaygroundState | None] = contextvars.ContextVar(
    "current_room", default=None
)


def get_state() -> PlaygroundState:
    room = current_room_var.get()
    if room is not None:
        return room
    return registry.get_or_create(PUBLIC_ROOM_ID)


def _resolve_stream_key(name: str, arguments: dict[str, Any]) -> tuple[str, str]:
    if name == "job.status_changed":
        job_id = arguments.get("job_id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise MCPError(INVALID_PARAMS, "job_id is required")
        return ("job.status_changed", job_id.strip())
    if name == "system.heartbeat":
        return ("system.heartbeat", "*")
    raise MCPError(EVENTS_NOT_FOUND, "NotFound", {"kind": "event"})


# ---------------------------------------------------------------------------
# SEP-3415 extension handlers
# ---------------------------------------------------------------------------


async def list_events(ctx: Any, params: PaginatedRequestParams) -> ListEventsResult:
    del ctx, params
    get_state().touch()
    return ListEventsResult(
        events=[
            EventDescriptor(
                name="job.status_changed",
                description=(
                    "Fires when a job transitions between stages (queued, running, done, failed)."
                ),
                delivery=["poll"],
                input_schema={
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string", "description": "Target job identifier"}
                    },
                    "required": ["job_id"],
                },
                payload_schema={
                    "type": "object",
                    "properties": {
                        "job_id": {"type": "string"},
                        "stage": {"type": "string"},
                        "detail": {"type": "string"},
                    },
                },
            ),
            EventDescriptor(
                name="system.heartbeat",
                description="Server heartbeat with no required arguments.",
                delivery=["poll"],
                input_schema={"type": "object", "properties": {}},
                payload_schema={
                    "type": "object",
                    "properties": {
                        "seq": {"type": "integer"},
                        "note": {"type": "string"},
                    },
                },
            ),
            EventDescriptor(
                name="webhook.only_alert",
                description="Webhook-only alert (not pollable; useful for testing client UI).",
                delivery=["webhook"],
                input_schema={"type": "object", "properties": {}},
            ),
        ]
    )


async def poll_events(ctx: Any, params: PollEventsRequestParams) -> PollEventsResult:
    del ctx
    state = get_state()
    state.touch()
    key = _resolve_stream_key(params.name, params.arguments)

    if key in state.forbidden_keys:
        state.notify({"kind": "fault", "reason": "forbidden", "name": params.name})
        raise MCPError(EVENTS_FORBIDDEN, "Subscription revoked by server")

    state.last_polled[key] = time.monotonic()

    history = state.streams.get(key, [])
    total = state.stream_totals.get(key, 0)
    # Sequence number of the last evicted event (0 when nothing has been trimmed).
    evicted_through = total - len(history)
    truncated = False

    if params.cursor is None:
        start = len(history)
    else:
        try:
            cursor_seq = int(params.cursor)
        except ValueError as exc:
            raise MCPError(INVALID_PARAMS, f"Invalid cursor: {params.cursor!r}") from exc
        if cursor_seq < 0:
            raise MCPError(INVALID_PARAMS, f"Invalid cursor: {params.cursor!r}")
        if cursor_seq < evicted_through or cursor_seq > total:
            truncated = True
            start = 0
        else:
            start = cursor_seq - evicted_through

    if key in state.pending_gaps and params.cursor is not None:
        state.pending_gaps.discard(key)
        truncated = True

    sliced = history[start:]
    if params.max_age_ms is not None and params.max_age_ms >= 0:
        cutoff = datetime.now(UTC) - timedelta(milliseconds=params.max_age_ms)
        filtered = [ev for ev in sliced if ev.timestamp >= cutoff]
        if len(filtered) < len(sliced):
            truncated = True
        sliced = filtered

    has_more = False
    next_cursor_seq = total
    if params.max_events is not None and params.max_events > 0 and len(sliced) > params.max_events:
        sliced = sliced[: params.max_events]
        has_more = True
        next_cursor_seq = sliced[-1].stream_seq

    result = PollEventsResult(
        events=[ev.to_occurrence() for ev in sliced],
        cursor=str(next_cursor_seq),
        truncated=truncated,
        has_more=has_more,
        next_poll_ms=state.poll_interval_ms,
    )

    state.poll_log.append(
        {
            "at": datetime.now(UTC).isoformat(),
            "name": params.name,
            "arguments": dict(params.arguments),
            "cursor_in": params.cursor,
            "cursor_out": result.cursor,
            "returned": len(result.events),
            "truncated": truncated,
            "has_more": has_more,
        }
    )
    if len(state.poll_log) > MAX_POLL_LOG:
        del state.poll_log[:-MAX_POLL_LOG]

    state.notify(
        {
            "kind": "fault" if truncated else "poll",
            "name": params.name,
            "filter_key": key[1],
            "returned": len(result.events),
            "truncated": truncated,
        }
    )

    return result


class PlaygroundEventsExtension(Extension):
    identifier = EVENTS_EXTENSION

    def methods(self) -> list[MethodBinding]:
        return [
            MethodBinding("events/list", PaginatedRequestParams, list_events),
            MethodBinding("events/poll", PollEventsRequestParams, poll_events),
        ]


mcp = MCPServer("events-playground", extensions=[PlaygroundEventsExtension()])


@mcp.tool()
def start_job(job_id: str, description: str = "training run") -> str:
    """Start a new background job and emit its initial queued event."""
    state = get_state()
    state.emit(
        "job.status_changed",
        job_id,
        {"job_id": job_id, "stage": "queued", "detail": description},
    )
    return (
        f"Started job {job_id} ({description}). "
        f"Subscribe to job.status_changed with job_id={job_id} to follow updates."
    )


@mcp.tool()
def get_job_details(job_id: str) -> dict[str, Any]:
    """Fetch recorded history and latest stage for a job."""
    state = get_state()
    state.touch()
    history = state.streams.get(("job.status_changed", job_id), [])
    return {
        "job_id": job_id,
        "stages": [
            {"event_id": ev.event_id, "timestamp": ev.timestamp.isoformat(), **ev.data}
            for ev in history
        ],
        "latest_stage": history[-1].data.get("stage", "unknown") if history else "unknown",
    }


@mcp.tool()
def advance_job(job_id: str, stage: str, detail: str = "") -> str:
    """Advance a job to a new stage, firing job.status_changed."""
    state = get_state()
    state.emit(
        "job.status_changed",
        job_id,
        {"job_id": job_id, "stage": stage, "detail": detail},
    )
    return f"Job {job_id} -> {stage}"


# ---------------------------------------------------------------------------
# HTTP control plane & browser UI
# ---------------------------------------------------------------------------


def _room_from_request(request: Request) -> PlaygroundState | None:
    raw_room = request.path_params.get("room_id")
    if raw_room is None:
        return registry.get_or_create(PUBLIC_ROOM_ID)
    valid = registry.validate_room_id(str(raw_room))
    if valid is None:
        return None
    return registry.get_or_create(valid)


def _invalid_room_response() -> JSONResponse:
    return JSONResponse(
        {"error": "room_id must be 3-32 lowercase letters, digits, or hyphens"},
        status_code=400,
    )


STATIC_DIR = Path(__file__).resolve().parent / "static"


async def http_health(request: Request) -> JSONResponse:
    del request
    return JSONResponse({"status": "ok"})


async def http_static_hf_logo(request: Request) -> Response:
    del request
    asset = STATIC_DIR / "hf-logo.svg"
    if not asset.is_file():
        return JSONResponse({"error": "asset not found"}, status_code=404)
    return FileResponse(
        asset,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=86400"},
    )


async def http_emit(request: Request) -> JSONResponse:
    state = _room_from_request(request)
    if state is None:
        return _invalid_room_response()
    body = await request.json()
    event_name = str(body.get("event_name") or "job.status_changed")
    filter_key = str(body.get("filter_key") or body.get("job_id") or "*")
    data = body.get("data")
    if not isinstance(data, dict):
        data = {
            "job_id": filter_key,
            "stage": str(body.get("stage") or "running"),
            "detail": str(body.get("detail") or ""),
        }
    record = state.emit(event_name, filter_key, data)
    return JSONResponse(record.to_dict())


async def http_gap(request: Request) -> JSONResponse:
    state = _room_from_request(request)
    if state is None:
        return _invalid_room_response()
    body = await request.json()
    key = (
        str(body.get("event_name") or "job.status_changed"),
        str(body.get("filter_key") or body.get("job_id") or "*"),
    )
    state.pending_gaps.add(key)
    state.notify({"kind": "fault", "reason": "gap_armed", "name": key[0]})
    return JSONResponse({"gap_armed": list(key)})


async def http_terminate(request: Request) -> JSONResponse:
    state = _room_from_request(request)
    if state is None:
        return _invalid_room_response()
    body = await request.json()
    key = (
        str(body.get("event_name") or "job.status_changed"),
        str(body.get("filter_key") or body.get("job_id") or "*"),
    )
    state.forbidden_keys.add(key)
    state.notify({"kind": "fault", "reason": "revoked", "name": key[0]})
    return JSONResponse({"forbidden": list(key)})


async def http_unterminate(request: Request) -> JSONResponse:
    state = _room_from_request(request)
    if state is None:
        return _invalid_room_response()
    body = await request.json()
    key = (
        str(body.get("event_name") or "job.status_changed"),
        str(body.get("filter_key") or body.get("job_id") or "*"),
    )
    state.forbidden_keys.discard(key)
    state.notify({"kind": "state"})
    return JSONResponse({"restored": list(key)})


async def http_config(request: Request) -> JSONResponse:
    state = _room_from_request(request)
    if state is None:
        return _invalid_room_response()
    body = await request.json()
    if "poll_interval_ms" in body:
        try:
            state.poll_interval_ms = max(250, min(int(body["poll_interval_ms"]), 300_000))
        except (TypeError, ValueError):
            return JSONResponse({"error": "poll_interval_ms must be an integer"}, status_code=400)
        state.notify({"kind": "state"})
    return JSONResponse({"poll_interval_ms": state.poll_interval_ms})


async def http_reset(request: Request) -> JSONResponse:
    state = _room_from_request(request)
    if state is None:
        return _invalid_room_response()
    state.reset()
    return JSONResponse({"reset": True, "room_id": state.room_id})


async def http_stats(request: Request) -> JSONResponse:
    state = _room_from_request(request)
    if state is None:
        return _invalid_room_response()
    return JSONResponse(
        {
            "room_id": state.room_id,
            "poll_interval_ms": state.poll_interval_ms,
            "streams": {f"{k[0]}:{k[1]}": len(v) for k, v in state.streams.items()},
            "active_streams": state.active_streams(),
            "pending_gaps": [f"{k[0]}:{k[1]}" for k in sorted(state.pending_gaps)],
            "forbidden": [f"{k[0]}:{k[1]}" for k in sorted(state.forbidden_keys)],
            "event_count": state.event_seq,
            "poll_count": len(state.poll_log),
            "recent_events": state.recent_events[-15:],
            "recent_polls": state.poll_log[-15:],
        }
    )


async def http_stream(request: Request) -> Response:
    """Server-Sent Events stream pushing real-time emit/poll pulses to the UI."""
    state = _room_from_request(request)
    if state is None:
        return _invalid_room_response()

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=32)
    state.subscribers.add(queue)

    async def event_generator() -> AsyncIterator[str]:
        try:
            yield f"data: {json.dumps({'kind': 'connected'})}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(item)}\n\n"
                except TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            state.subscribers.discard(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="playground-room" content="__ROOM_ID__">
  <title>MCP Events Test Lab</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      color-scheme: light;
      --bg: #eceef2;
      --surface: #ffffff;
      --surface-subtle: #f5f6f9;
      --border: #d4dae3;
      --border-strong: #b4becd;
      --ink: #12161f;
      --muted: #4c5667;
      --accent: #1849d6;
      --accent-hover: #1239aa;
      --warm: #c84323;
      --warm-soft: #fdf1ee;
      --ok: #16794b;
      --scope-bg: #0d121c;
      --scope-border: #202a3d;
      --scope-ink: #e5ebf5;
      --scope-muted: #8c9bb3;
      --scope-emit: #3b82f6;
      --scope-poll: #34d399;
      --scope-fault: #f97316;
      --font-sans: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --font-mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
      --text-heading: 1.125rem;
      --text-body: 0.9375rem;
      --text-sm: 0.8125rem;
      --radius: 10px;
      --radius-sm: 6px;
      --ease: cubic-bezier(0.22, 1, 0.36, 1);
    }

    * { box-sizing: border-box; }

    html, body {
      overflow-x: clip;
    }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: var(--font-sans);
      font-size: var(--text-body);
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }

    a {
      color: var(--ink);
      text-decoration: none;
    }

    a:hover {
      color: var(--accent);
    }

    .topbar {
      position: sticky;
      top: 0;
      z-index: 20;
      background: rgba(236, 238, 242, 0.92);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--border);
    }

    .topbar-inner {
      max-width: 1120px;
      margin: 0 auto;
      padding: 0.8rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
    }

    .brand {
      display: flex;
      align-items: baseline;
      gap: 0.75rem;
      min-width: 0;
    }

    .brand-title {
      font-size: 1.125rem;
      font-weight: 600;
      letter-spacing: -0.02em;
      white-space: nowrap;
    }

    .brand-room {
      font-family: var(--font-mono);
      font-size: var(--text-sm);
      color: var(--muted);
      font-variant-numeric: tabular-nums;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .nav {
      display: flex;
      align-items: center;
      gap: 1.25rem;
      font-size: var(--text-sm);
      font-weight: 500;
      white-space: nowrap;
    }

    .nav a {
      color: var(--muted);
      transition: color 160ms var(--ease);
    }

    .nav a:hover {
      color: var(--ink);
    }

    .page {
      max-width: 1120px;
      margin: 0 auto;
      padding: 1.5rem 1.5rem 3rem;
      display: flex;
      flex-direction: column;
      gap: 1.35rem;
    }

    /* Opening workspace band */
    .opening {
      display: grid;
      grid-template-columns: minmax(0, 1.15fr) minmax(0, 0.85fr);
      gap: 1.5rem;
      align-items: stretch;
    }

    .opening > * {
      min-width: 0;
    }

    .opening-left {
      display: flex;
      flex-direction: column;
      justify-content: center;
    }

    h1 {
      margin: 0 0 0.5rem;
      font-size: clamp(1.75rem, 3vw, 2.35rem);
      font-weight: 600;
      line-height: 1.08;
      letter-spacing: -0.03em;
      color: var(--ink);
    }

    .opening-lead {
      margin: 0 0 1.1rem;
      max-width: 54ch;
      color: var(--muted);
      font-size: var(--text-body);
    }

    .connect-panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 0.85rem 1rem;
      display: flex;
      flex-direction: column;
      gap: 0.65rem;
    }

    .connect-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.65rem;
      flex-wrap: wrap;
    }

    .segmented {
      display: inline-flex;
      background: var(--surface-subtle);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 2px;
      gap: 2px;
    }

    .segmented button,
    .segmented label {
      position: relative;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      margin: 0;
      padding: 0.3rem 0.62rem;
      border: 0;
      border-radius: 4px;
      background: transparent;
      color: var(--muted);
      font-family: var(--font-sans);
      font-size: var(--text-sm);
      font-weight: 500;
      cursor: pointer;
      white-space: nowrap;
      transition: background-color 150ms var(--ease), color 150ms var(--ease);
    }

    .segmented input[type="radio"] {
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }

    .segmented button[aria-selected="true"],
    .segmented label:has(input:checked) {
      background: var(--surface);
      color: var(--ink);
      box-shadow: 0 1px 2px rgba(18, 22, 31, 0.08);
    }

    .command-row {
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }

    .command-code {
      flex: 1;
      min-width: 0;
      padding: 0.54rem 0.72rem;
      background: var(--surface-subtle);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      font-family: var(--font-mono);
      font-size: var(--text-sm);
      color: var(--ink);
      overflow-x: auto;
      white-space: nowrap;
    }

    .btn-primary {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0.5rem 0.95rem;
      border: 1px solid var(--accent);
      border-radius: var(--radius-sm);
      background: var(--accent);
      color: #ffffff;
      font-family: var(--font-sans);
      font-size: var(--text-sm);
      font-weight: 500;
      cursor: pointer;
      white-space: nowrap;
      transition: background-color 150ms var(--ease), border-color 150ms var(--ease), transform 120ms var(--ease);
    }

    .btn-primary:hover {
      background: var(--accent-hover);
      border-color: var(--accent-hover);
    }

    .btn-primary:active {
      transform: translateY(1px);
    }

    .btn-secondary {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0.46rem 0.82rem;
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-sm);
      background: var(--surface);
      color: var(--ink);
      font-family: var(--font-sans);
      font-size: var(--text-sm);
      font-weight: 500;
      cursor: pointer;
      white-space: nowrap;
      transition: background-color 150ms var(--ease), border-color 150ms var(--ease);
    }

    .btn-secondary:hover {
      background: var(--surface-subtle);
      border-color: var(--ink);
    }

    .btn-waiting {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0.46rem 0.82rem;
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      background: var(--surface-subtle);
      color: var(--muted);
      font-family: var(--font-sans);
      font-size: var(--text-sm);
      font-weight: 500;
      cursor: pointer;
      white-space: nowrap;
      transition: background-color 160ms var(--ease), border-color 160ms var(--ease), color 160ms var(--ease);
    }

    .btn-waiting:hover {
      border-color: var(--border-strong);
      color: var(--ink);
    }

    .btn-danger {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      padding: 0.46rem 0.82rem;
      border: 1px solid #e6b4a8;
      border-radius: var(--radius-sm);
      background: var(--warm-soft);
      color: var(--warm);
      font-family: var(--font-sans);
      font-size: var(--text-sm);
      font-weight: 500;
      cursor: pointer;
      white-space: nowrap;
      transition: background-color 150ms var(--ease), border-color 150ms var(--ease);
    }

    .btn-danger:hover {
      background: #fae2dc;
      border-color: var(--warm);
    }

    .toast {
      min-height: 1.2rem;
      font-size: var(--text-sm);
      color: var(--ok);
      font-weight: 500;
    }

    /* Compact live hardware-styled oscilloscope monitor */
    .scope-unit {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 0.85rem;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 0.75rem;
    }

    .scope-screen {
      background: var(--scope-bg);
      border: 1px solid var(--scope-border);
      border-radius: var(--radius-sm);
      padding: 0.55rem 0.65rem 0.45rem;
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
    }

    .scope-screen-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.5rem;
      font-family: var(--font-mono);
      font-size: var(--text-sm);
      color: var(--scope-muted);
    }

    .scope-leds {
      display: inline-flex;
      align-items: center;
      gap: 0.85rem;
    }

    .led-item {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      color: var(--scope-muted);
      transition: color 180ms var(--ease);
    }

    .led-dot {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: #28354d;
      transition: background-color 160ms var(--ease), box-shadow 160ms var(--ease);
    }

    .led-item.lit-emit {
      color: var(--scope-ink);
    }

    .led-item.lit-emit .led-dot {
      background: var(--scope-emit);
      box-shadow: 0 0 8px rgba(59, 130, 246, 0.9);
    }

    .led-item.lit-poll {
      color: var(--scope-ink);
    }

    .led-item.lit-poll .led-dot {
      background: var(--scope-poll);
      box-shadow: 0 0 8px rgba(52, 211, 153, 0.9);
    }

    .led-item.lit-fault {
      color: var(--scope-ink);
    }

    .led-item.lit-fault .led-dot {
      background: var(--scope-fault);
      box-shadow: 0 0 8px rgba(249, 115, 22, 0.9);
    }

    .scope-canvas {
      width: 100%;
      height: 86px;
      display: block;
    }

    .scope-readout {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.5rem;
      font-variant-numeric: tabular-nums;
    }

    .readout-label {
      display: block;
      font-size: var(--text-sm);
      color: var(--muted);
    }

    .readout-value {
      display: block;
      font-family: var(--font-mono);
      font-size: var(--text-body);
      font-weight: 500;
      color: var(--ink);
    }

    .scope-status-line {
      font-family: var(--font-mono);
      font-size: var(--text-sm);
      color: var(--muted);
      padding-top: 0.55rem;
      border-top: 1px solid var(--border);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    /* Sections */
    .section {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.25rem 1.4rem;
    }

    .section-head {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 1rem;
      flex-wrap: wrap;
      margin-bottom: 0.85rem;
    }

    h2 {
      margin: 0;
      font-size: var(--text-heading);
      font-weight: 600;
      letter-spacing: -0.02em;
      color: var(--ink);
    }

    .section-note {
      margin: 0;
      font-size: var(--text-sm);
      color: var(--muted);
    }

    /* Prompts list */
    .prompt-list {
      display: flex;
      flex-direction: column;
    }

    .prompt-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 0.75rem 1.25rem;
      align-items: center;
      padding: 0.9rem 0;
      border-top: 1px solid var(--border);
    }

    .prompt-row:first-child {
      padding-top: 0.2rem;
      border-top: 0;
    }

    .prompt-row:last-child {
      padding-bottom: 0;
    }

    .prompt-row > div {
      min-width: 0;
    }

    .prompt-title {
      margin: 0 0 0.2rem;
      font-size: var(--text-body);
      font-weight: 600;
      color: var(--ink);
    }

    .prompt-input {
      margin: 0 0 0.3rem;
      font-family: var(--font-mono);
      font-size: var(--text-sm);
      line-height: 1.45;
      color: var(--ink);
      background: var(--surface-subtle);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 0.45rem 0.68rem;
      display: block;
      width: 100%;
      resize: vertical;
      min-height: 2.35rem;
      transition: border-color 150ms var(--ease), background-color 150ms var(--ease);
    }

    .prompt-input:focus {
      outline: none;
      border-color: var(--accent);
      background: var(--surface);
    }

    .tui-group {
      margin-top: 1.15rem;
      padding-top: 1rem;
      border-top: 1px solid var(--border);
    }

    .tui-group-head {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 0.75rem;
      margin-bottom: 0.65rem;
    }

    .tui-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.55rem 1.5rem;
    }

    .tui-cmd {
      display: flex;
      align-items: baseline;
      gap: 0.65rem;
      font-size: var(--text-sm);
      color: var(--muted);
      padding: 0.35rem 0.5rem;
      border-radius: var(--radius-sm);
      background: var(--surface-subtle);
      border: 1px solid var(--border);
      cursor: pointer;
      text-align: left;
      font-family: var(--font-sans);
      transition: border-color 150ms var(--ease), background-color 150ms var(--ease);
    }

    .tui-cmd:hover {
      border-color: var(--border-strong);
      background: var(--surface);
    }

    .tui-cmd code {
      font-family: var(--font-mono);
      color: var(--ink);
      font-weight: 500;
      white-space: nowrap;
    }

    .hf-banner {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1rem 1.4rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      flex-wrap: wrap;
    }

    .hf-banner-left {
      display: flex;
      align-items: center;
      gap: 0.85rem;
      min-width: 0;
    }

    .hf-logo {
      flex: 0 0 auto;
      width: 38px;
      height: 38px;
      display: block;
    }

    .hf-banner-title {
      margin: 0;
      font-size: var(--text-body);
      font-weight: 600;
      color: var(--ink);
    }

    .hf-banner-sub {
      margin: 0.1rem 0 0;
      font-size: var(--text-sm);
      color: var(--muted);
    }

    .hf-banner-sub a {
      color: var(--accent);
      font-family: var(--font-mono);
      font-weight: 500;
    }

    .hf-banner-sub a:hover {
      text-decoration: underline;
    }

    .prompt-step {
      margin: 0;
      font-size: var(--text-sm);
      color: var(--muted);
    }

    .prompt-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      align-items: center;
      justify-content: flex-end;
    }

    /* Control deck */
    .deck-grid {
      display: grid;
      grid-template-columns: minmax(0, 0.85fr) minmax(0, 1.15fr);
      gap: 1.5rem;
    }

    .deck-col {
      display: flex;
      flex-direction: column;
      gap: 0.7rem;
    }

    .deck-col + .deck-col {
      padding-left: 1.5rem;
      border-left: 1px solid var(--border);
    }

    h3 {
      margin: 0;
      font-size: var(--text-body);
      font-weight: 600;
      color: var(--ink);
    }

    .mono-sub {
      font-family: var(--font-mono);
      font-size: var(--text-sm);
      font-weight: 400;
      color: var(--muted);
      margin-left: 0.35rem;
    }

    .field-row {
      display: flex;
      flex-wrap: wrap;
      gap: 0.65rem;
      align-items: flex-end;
    }

    .field {
      display: flex;
      flex-direction: column;
      gap: 0.25rem;
      flex: 1;
      min-width: 140px;
    }

    .field-compact {
      flex: 0 0 auto;
      min-width: 110px;
    }

    .field label {
      font-size: var(--text-sm);
      color: var(--muted);
      font-weight: 500;
    }

    input[type="text"],
    input[type="number"] {
      width: 100%;
      padding: 0.46rem 0.62rem;
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-sm);
      background: var(--surface);
      color: var(--ink);
      font-family: var(--font-sans);
      font-size: var(--text-sm);
      transition: border-color 150ms var(--ease);
    }

    input[type="text"]:focus,
    input[type="number"]:focus {
      outline: none;
      border-color: var(--accent);
    }

    .action-row {
      display: flex;
      flex-wrap: wrap;
      gap: 0.6rem;
      align-items: center;
      margin-top: 0.1rem;
    }

    .fault-bar {
      margin-top: 1.15rem;
      padding-top: 1rem;
      border-top: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }

    .fault-controls {
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem 1rem;
      align-items: flex-end;
      justify-content: space-between;
    }

    .fault-group {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      align-items: flex-end;
    }

    .fault-notes {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 0.5rem 1.5rem;
      font-size: var(--text-sm);
      color: var(--muted);
    }

    .fault-notes p {
      margin: 0;
    }

    .fault-notes strong {
      color: var(--ink);
      font-weight: 500;
    }

    .fault-notes code {
      font-family: var(--font-mono);
      color: var(--ink);
    }

    /* Activity inspector */
    .inspector-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 1.5rem;
    }

    .inspector-col + .inspector-col {
      padding-left: 1.5rem;
      border-left: 1px solid var(--border);
    }

    .log-list {
      margin-top: 0.6rem;
      display: flex;
      flex-direction: column;
    }

    .log-item {
      padding: 0.55rem 0;
      border-top: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      gap: 0.2rem;
      font-size: var(--text-sm);
    }

    .log-item:first-child {
      border-top: 0;
      padding-top: 0.15rem;
    }

    .log-meta {
      display: flex;
      justify-content: space-between;
      gap: 0.5rem;
      font-family: var(--font-mono);
      font-variant-numeric: tabular-nums;
      color: var(--ink);
    }

    .log-time {
      color: var(--muted);
    }

    .log-detail {
      color: var(--muted);
      font-family: var(--font-mono);
      word-break: break-word;
    }

    .status-warn {
      color: var(--warm);
      font-weight: 500;
    }

    .status-ok {
      color: var(--ok);
      font-weight: 500;
    }

    .empty-state {
      margin: 0.65rem 0 0;
      font-size: var(--text-sm);
      color: var(--muted);
    }

    pre.raw-json {
      margin: 0.6rem 0 0;
      padding: 0.85rem;
      border-radius: var(--radius-sm);
      background: var(--scope-bg);
      color: var(--scope-ink);
      font-family: var(--font-mono);
      font-size: var(--text-sm);
      overflow: auto;
      max-height: 360px;
    }

    .hidden {
      display: none !important;
    }

    .footer {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 0.75rem;
      font-size: var(--text-sm);
      color: var(--muted);
      padding-top: 0.2rem;
    }

    .footer code {
      font-family: var(--font-mono);
      color: var(--ink);
    }

    @media (max-width: 880px) {
      .opening {
        grid-template-columns: 1fr;
        gap: 1.1rem;
      }
      .deck-grid,
      .inspector-grid,
      .fault-notes,
      .tui-grid {
        grid-template-columns: 1fr;
      }
      .deck-col + .deck-col,
      .inspector-col + .inspector-col {
        padding-left: 0;
        padding-top: 1.15rem;
        border-left: 0;
        border-top: 1px solid var(--border);
      }
    }

    @media (max-width: 640px) {
      .topbar-inner {
        padding: 0.75rem 1rem;
        flex-wrap: wrap;
        gap: 0.4rem 1rem;
      }
      .brand-room {
        display: none;
      }
      .nav {
        gap: 1rem;
      }
      .page {
        padding: 1.15rem 1rem 2.25rem;
      }
      .section {
        padding: 1.05rem 1rem;
      }
      .prompt-row {
        grid-template-columns: 1fr;
      }
      .prompt-actions {
        justify-content: flex-start;
      }
      .command-row {
        flex-direction: column;
        align-items: stretch;
      }
      .segmented {
        max-width: 100%;
        overflow-x: auto;
      }
    }

    @media (prefers-reduced-motion: reduce) {
      *,
      *::before,
      *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
      }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="topbar-inner">
      <div class="brand">
        <span class="brand-title">MCP Events Test Lab</span>
        <span class="brand-room" id="header-room"></span>
      </div>
      <nav class="nav" aria-label="Playground links">
        <a href="/new">New room</a>
        <a href="https://modelcontextprotocol.io/seps/3415-events-extension" target="_blank" rel="noreferrer">SEP-3415 spec</a>
        <a href="https://fast-agent.ai" target="_blank" rel="noreferrer">fast-agent</a>
      </nav>
    </div>
  </header>

  <main class="page">
    <section class="opening">
      <div class="opening-left">
        <h1>MCP Events Test Lab</h1>
        <p class="opening-lead">
          Trigger server events in your isolated room and watch fast-agent wake from idle, catch up after gaps, and handle revoked streams.
        </p>

        <div class="connect-panel">
          <div class="connect-head">
            <div class="segmented" role="tablist" aria-label="Connection command format">
              <button type="button" role="tab" aria-selected="true" data-cmd="uvx" onclick="setCmdMode('uvx')">uvx</button>
              <button type="button" role="tab" aria-selected="false" data-cmd="go" onclick="setCmdMode('go')">fast-agent go</button>
              <button type="button" role="tab" aria-selected="false" data-cmd="connect" onclick="setCmdMode('connect')">/mcp connect</button>
              <button type="button" role="tab" aria-selected="false" data-cmd="url" onclick="setCmdMode('url')">URL</button>
            </div>
            <span class="toast" id="toast" role="status" aria-live="polite"></span>
          </div>
          <div class="command-row">
            <code class="command-code" id="cmd-output"></code>
            <button type="button" class="btn-primary" onclick="copyCurrentCommand()">Copy</button>
          </div>
        </div>
      </div>

      <aside class="scope-unit" aria-label="Live signal monitor">
        <div class="scope-screen">
          <div class="scope-screen-head">
            <span>Signal monitor</span>
            <div class="scope-leds">
              <span class="led-item" id="led-emit"><span class="led-dot"></span>Emit</span>
              <span class="led-item" id="led-poll"><span class="led-dot"></span>Poll</span>
            </div>
          </div>
          <svg class="scope-canvas" id="scope-svg" viewBox="0 0 420 86" role="img" aria-label="Live waveform showing emitted events in blue and incoming client polls in green">
            <!-- Subtle reticle grid -->
            <line x1="0" y1="22" x2="420" y2="22" stroke="#182235" stroke-width="1" />
            <line x1="0" y1="44" x2="420" y2="44" stroke="#1f2d45" stroke-width="1" />
            <line x1="0" y1="66" x2="420" y2="66" stroke="#182235" stroke-width="1" />
            <line x1="105" y1="0" x2="105" y2="86" stroke="#182235" stroke-width="1" />
            <line x1="210" y1="0" x2="210" y2="86" stroke="#182235" stroke-width="1" />
            <line x1="315" y1="0" x2="315" y2="86" stroke="#182235" stroke-width="1" />
            <!-- Traces updated live -->
            <polyline id="trace-emit" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linejoin="round" points="0,44 420,44" />
            <polyline id="trace-poll" fill="none" stroke="#34d399" stroke-width="1.8" stroke-linejoin="round" points="0,66 420,66" />
          </svg>
        </div>

        <div class="scope-readout">
          <div>
            <span class="readout-label">Events</span>
            <span class="readout-value" id="metric-events">0</span>
          </div>
          <div>
            <span class="readout-label">Polls</span>
            <span class="readout-value" id="metric-polls">0</span>
          </div>
          <div>
            <span class="readout-label">nextPollMs</span>
            <span class="readout-value" id="metric-interval">2000 ms</span>
          </div>
        </div>

        <div class="scope-status-line" id="scope-status">Waiting for a client to poll this room</div>
      </aside>
    </section>

    <section class="section">
      <div class="section-head">
        <h2>Prompts to try in fast-agent</h2>
        <p class="section-note">Trigger buttons activate as soon as fast-agent starts polling the matching stream.</p>
      </div>

      <div class="prompt-list">
        <div class="prompt-row">
          <div>
            <p class="prompt-title">Wake on heartbeat</p>
            <textarea class="prompt-input" spellcheck="false" id="prompt-1" rows="1" aria-label="Wake on heartbeat prompt">Register for the heartbeat and tell me an interesting fact every time it triggers.</textarea>
            <p class="prompt-step" id="hint-hb">Waiting for a system.heartbeat poll, or click Send heartbeat anytime. Edit the prompt to ask for one trigger (subscribe once) if you want it to stop after the first event.</p>
          </div>
          <div class="prompt-actions">
            <button type="button" class="btn-secondary" onclick="copyPrompt('prompt-1')">Copy prompt</button>
            <button type="button" class="btn-waiting" id="trigger-hb" onclick="emitHB()">Send heartbeat</button>
          </div>
        </div>

        <div class="prompt-row">
          <div>
            <p class="prompt-title">Follow a job to completion</p>
            <textarea class="prompt-input" spellcheck="false" id="prompt-2" rows="2" aria-label="Follow a job prompt">Start job-1 for "nightly fine-tune", subscribe to job.status_changed for it, and summarize the run once it finishes.</textarea>
            <p class="prompt-step" id="hint-job">Waiting for a job.status_changed poll, then emit running and done.</p>
          </div>
          <div class="prompt-actions">
            <button type="button" class="btn-secondary" onclick="copyPrompt('prompt-2')">Copy prompt</button>
            <button type="button" class="btn-waiting" id="trigger-job-run" onclick="emitQuickJob('running')">Emit running</button>
            <button type="button" class="btn-waiting" id="trigger-job-done" onclick="emitQuickJob('done')">Emit done</button>
          </div>
        </div>

        <div class="prompt-row">
          <div>
            <p class="prompt-title">Test gaps and revocation</p>
            <textarea class="prompt-input" spellcheck="false" id="prompt-3" rows="2" aria-label="Recover from gap prompt">Watch job-1 and if the server reports truncated events, fetch full job details to catch up.</textarea>
            <p class="prompt-step">Arm truncated gap sets truncated: true on the next poll; Revoke job returns -32024 Forbidden so fast-agent ends the subscription.</p>
          </div>
          <div class="prompt-actions">
            <button type="button" class="btn-secondary" onclick="copyPrompt('prompt-3')">Copy prompt</button>
            <button type="button" class="btn-waiting" id="trigger-gap" onclick="armGap()">Arm truncated gap</button>
            <button type="button" class="btn-waiting" id="trigger-revoke" onclick="armTerm()">Revoke job</button>
          </div>
        </div>

        <div class="prompt-row">
          <div>
            <p class="prompt-title">Buffer events without waking</p>
            <textarea class="prompt-input" spellcheck="false" id="prompt-4" rows="1" aria-label="Passive buffering prompt">Subscribe passively (wake=false) to system.heartbeat and hold updates until I ask.</textarea>
            <p class="prompt-step">Send a few heartbeats to see them queue in the toolbar without waking the agent. Your next turn automatically includes all queued events (just send any message, like "Any updates?").</p>
          </div>
          <div class="prompt-actions">
            <button type="button" class="btn-secondary" onclick="copyPrompt('prompt-4')">Copy prompt</button>
            <button type="button" class="btn-waiting" id="trigger-hb-passive" onclick="emitHB()">Send heartbeat</button>
          </div>
        </div>
      </div>

      <div class="tui-group" aria-label="fast-agent commands">
        <div class="tui-group-head">
          <h3>fast-agent commands</h3>
          <span class="section-note">Click any command to copy.</span>
        </div>
        <div class="tui-grid">
          <button type="button" class="tui-cmd" onclick="copyText('/events', 'Copied /events')">
            <code>/events</code>
            <span>Inspect active, completed, and revoked subscriptions</span>
          </button>
          <button type="button" class="tui-cmd" onclick="copyText('/events select', 'Copied /events select')">
            <code>/events select</code>
            <span>Interactive picker to subscribe or unsubscribe</span>
          </button>
          <button type="button" class="tui-cmd" onclick="copyText('/events poll ', 'Copied /events poll')">
            <code>/events poll &lt;id&gt;</code>
            <span>Force an immediate poll without waiting</span>
          </button>
          <button type="button" class="tui-cmd" onclick="copyText('/events unsubscribe all', 'Copied /events unsubscribe all')">
            <code>/events unsubscribe all</code>
            <span>Stop all active polling loops</span>
          </button>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <h2>Event control deck</h2>
        <p class="section-note">Custom payloads, poll interval, and fault injection for this room.</p>
      </div>

      <div class="deck-grid">
        <div class="deck-col">
          <h3>System heartbeat <span class="mono-sub">system.heartbeat</span></h3>
          <div class="field">
            <label for="hb-note">Optional note</label>
            <input id="hb-note" type="text" value="manual pulse" placeholder="manual pulse">
          </div>
          <div class="action-row">
            <button type="button" class="btn-primary" onclick="emitHB()">Send heartbeat</button>
          </div>
        </div>

        <div class="deck-col">
          <h3>Job lifecycle <span class="mono-sub">job.status_changed</span></h3>
          <div class="field-row">
            <div class="field field-compact">
              <label for="job">job_id</label>
              <input id="job" type="text" value="job-1">
            </div>
            <div class="field">
              <label for="detail">detail</label>
              <input id="detail" type="text" value="manual trigger">
            </div>
          </div>
          <div class="action-row">
            <div class="segmented" role="radiogroup" aria-label="Job stage">
              <label><input type="radio" name="stage" value="queued">queued</label>
              <label><input type="radio" name="stage" value="running" checked>running</label>
              <label><input type="radio" name="stage" value="done">done</label>
              <label><input type="radio" name="stage" value="failed">failed</label>
            </div>
            <button type="button" class="btn-primary" onclick="emitSelectedJob()">Emit stage</button>
          </div>
        </div>
      </div>

      <div class="fault-bar">
        <div class="fault-controls">
          <div class="fault-group">
            <div class="field field-compact">
              <label for="poll-ms">nextPollMs</label>
              <input id="poll-ms" type="number" value="2000" min="250" max="300000" step="250">
            </div>
            <button type="button" class="btn-secondary" onclick="setPollMs()">Apply</button>
          </div>

          <div class="fault-group">
            <button type="button" class="btn-secondary" onclick="armGap()">Arm truncated gap</button>
            <button type="button" class="btn-danger" onclick="armTerm()">Revoke job</button>
            <button type="button" class="btn-secondary" onclick="unTerm()">Restore job</button>
            <button type="button" class="btn-danger" onclick="resetAll()">Reset room</button>
          </div>
        </div>

        <div class="fault-notes">
          <p><strong>Arm truncated gap:</strong> marks the next <code>events/poll</code> response with <code>truncated: true</code>, simulating events that aged out of the server buffer so the agent knows to re-fetch state via <code>get_job_details</code>.</p>
          <p><strong>Revoke job:</strong> returns <code>-32024 Forbidden</code> on the next poll for this job so fast-agent terminates the subscription cleanly.</p>
        </div>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <h2>Room activity</h2>
        <div class="segmented" role="tablist" aria-label="Activity view mode">
          <button type="button" role="tab" aria-selected="true" data-view="formatted" onclick="setViewMode('formatted')">Formatted log</button>
          <button type="button" role="tab" aria-selected="false" data-view="raw" onclick="setViewMode('raw')">Raw JSON</button>
        </div>
      </div>

      <div class="inspector-grid" id="view-formatted">
        <div class="inspector-col">
          <h3>Recent events</h3>
          <div class="log-list" id="events-list">
            <p class="empty-state">No events emitted in this room yet.</p>
          </div>
        </div>
        <div class="inspector-col">
          <h3>Recent polls</h3>
          <div class="log-list" id="polls-list">
            <p class="empty-state">Waiting for a client to call events/poll.</p>
          </div>
        </div>
      </div>

      <div class="inspector-grid hidden" id="view-raw">
        <div class="inspector-col">
          <h3>Streams and events</h3>
          <pre class="raw-json" id="events-raw">Loading...</pre>
        </div>
        <div class="inspector-col">
          <h3>Poll log</h3>
          <pre class="raw-json" id="polls-raw">Loading...</pre>
        </div>
      </div>
    </section>

    <section class="hf-banner" aria-label="Hugging Face MCP Server">
      <div class="hf-banner-left">
        <img class="hf-logo" src="/static/hf-logo.svg" alt="Hugging Face logo" width="38" height="38">
        <div>
          <p class="hf-banner-title">Try with HF Jobs on the Hugging Face MCP Server</p>
          <p class="hf-banner-sub">Connect to <a href="https://hf.co/mcp" target="_blank" rel="noreferrer">https://hf.co/mcp</a> to subscribe to live HF Jobs events in fast-agent.</p>
        </div>
      </div>
      <a class="btn-secondary" href="https://hf.co/mcp" target="_blank" rel="noreferrer">Open hf.co/mcp</a>
    </section>

    <footer class="footer">
      <span>Isolated room <code id="footer-room"></code> expires after 4 hours of inactivity.</span>
      <span>Manual TUI commands: <code>/events</code> and <code>/events select</code></span>
    </footer>
  </main>

  <script>
    const roomMeta = document.querySelector('meta[name="playground-room"]');
    const roomId = roomMeta ? roomMeta.content : 'public';
    const apiPrefix = roomId === 'public' ? '' : '/r/' + encodeURIComponent(roomId);
    const mcpUrl = window.location.origin + apiPrefix + '/mcp';

    let cmdMode = 'uvx';
    let activeJobId = null;

    document.getElementById('header-room').textContent = 'room: ' + roomId;
    document.getElementById('footer-room').textContent = roomId;

    // -----------------------------------------------------------------------
    // Live oscilloscope waveform (scrolling SVG trace)
    // -----------------------------------------------------------------------
    const SAMPLES = 84;
    const emitBuf = new Array(SAMPLES).fill(0);
    const pollBuf = new Array(SAMPLES).fill(0);
    const emitQueue = [];
    const pollQueue = [];
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function flashLed(kind) {
      const el = document.getElementById(kind === 'emit' ? 'led-emit' : 'led-poll');
      if (!el) return;
      const cls = kind === 'emit' ? 'lit-emit' : (kind === 'fault' ? 'lit-fault' : 'lit-poll');
      el.classList.remove('lit-emit', 'lit-poll', 'lit-fault');
      void el.offsetWidth;
      el.classList.add(cls);
      setTimeout(() => el.classList.remove(cls), 520);
    }

    function pushPulse(kind) {
      if (kind === 'emit') {
        emitQueue.push(1, 1, 1, 1, 0);
        flashLed('emit');
      } else if (kind === 'fault') {
        pollQueue.push(-1, 1, -1, 0);
        flashLed('fault');
      } else {
        pollQueue.push(1, 1, 0);
        flashLed('poll');
      }
      if (reducedMotion) renderWaveforms();
    }

    function renderWaveforms() {
      const step = 420 / (SAMPLES - 1);
      const emitPts = emitBuf.map((v, i) => (i * step).toFixed(1) + ',' + (44 - v * 22).toFixed(1)).join(' ');
      const pollPts = pollBuf.map((v, i) => (i * step).toFixed(1) + ',' + (66 - v * 16).toFixed(1)).join(' ');
      document.getElementById('trace-emit').setAttribute('points', emitPts);
      document.getElementById('trace-poll').setAttribute('points', pollPts);
    }

    if (!reducedMotion) {
      setInterval(() => {
        emitBuf.shift();
        emitBuf.push(emitQueue.length ? emitQueue.shift() : 0);
        pollBuf.shift();
        pollBuf.push(pollQueue.length ? pollQueue.shift() : 0);
        renderWaveforms();
      }, 65);
    }

    // -----------------------------------------------------------------------
    // Connection command bar
    // -----------------------------------------------------------------------
    function getCommandText(mode) {
      if (mode === 'uvx') return 'uvx fast-agent-mcp@latest go --url ' + mcpUrl;
      if (mode === 'go') return 'fast-agent go --url ' + mcpUrl;
      if (mode === 'connect') return '/mcp connect ' + mcpUrl + ' --name playground';
      return mcpUrl;
    }

    function setCmdMode(mode) {
      cmdMode = mode;
      document.querySelectorAll('[data-cmd]').forEach((btn) => {
        btn.setAttribute('aria-selected', btn.dataset.cmd === mode ? 'true' : 'false');
      });
      document.getElementById('cmd-output').textContent = getCommandText(mode);
    }

    function setViewMode(mode) {
      document.querySelectorAll('[data-view]').forEach((btn) => {
        btn.setAttribute('aria-selected', btn.dataset.view === mode ? 'true' : 'false');
      });
      document.getElementById('view-formatted').classList.toggle('hidden', mode !== 'formatted');
      document.getElementById('view-raw').classList.toggle('hidden', mode !== 'raw');
    }

    function toast(msg) {
      const el = document.getElementById('toast');
      el.textContent = msg;
      setTimeout(() => { if (el.textContent === msg) el.textContent = ''; }, 2600);
    }

    async function copyText(text, label) {
      try {
        await navigator.clipboard.writeText(text);
        toast(label || 'Copied to clipboard');
      } catch {
        toast('Select and copy manually');
      }
    }

    function copyCurrentCommand() {
      copyText(getCommandText(cmdMode), 'Command copied');
    }

    function copyPrompt(id) {
      const el = document.getElementById(id);
      if (el) {
        const text = (typeof el.value === 'string' ? el.value : el.textContent).trim();
        copyText(text, 'Prompt copied');
      }
    }

    // -----------------------------------------------------------------------
    // Control plane actions
    // -----------------------------------------------------------------------
    async function post(path, body = {}) {
      const r = await fetch(apiPrefix + path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await r.json();
      await refresh();
      return data;
    }

    function jobId() { return document.getElementById('job').value.trim() || 'job-1'; }
    function targetJobId() { return activeJobId || jobId(); }
    function detail() { return document.getElementById('detail').value; }
    function selectedStage() {
      const checked = document.querySelector('input[name="stage"]:checked');
      return checked ? checked.value : 'running';
    }

    async function emitQuickJob(stage) {
      const target = targetJobId();
      const res = await post('/emit', {
        event_name: 'job.status_changed',
        job_id: target,
        stage,
        detail: stage === 'done' ? 'completed cleanly' : 'in progress',
      });
      toast('Emitted ' + res.event_id + ' (' + target + ': ' + stage + ')');
    }

    async function emitSelectedJob() {
      const stage = selectedStage();
      const res = await post('/emit', {
        event_name: 'job.status_changed',
        job_id: jobId(),
        stage,
        detail: detail(),
      });
      toast('Emitted ' + res.event_id + ' (' + stage + ')');
    }

    async function emitHB() {
      const note = document.getElementById('hb-note').value;
      const res = await post('/emit', {
        event_name: 'system.heartbeat',
        filter_key: '*',
        data: { seq: Date.now(), note },
      });
      toast('Emitted ' + res.event_id + ' (system.heartbeat)');
    }

    async function armGap() {
      const target = targetJobId();
      await post('/emit-gap', { event_name: 'job.status_changed', job_id: target });
      toast('Armed truncated gap for ' + target);
    }

    async function armTerm() {
      const target = targetJobId();
      await post('/terminate', { event_name: 'job.status_changed', job_id: target });
      toast('Revoked subscription for ' + target);
    }

    async function unTerm() {
      const target = targetJobId();
      await post('/unterminate', { event_name: 'job.status_changed', job_id: target });
      toast('Restored subscription for ' + target);
    }

    async function setPollMs() {
      const ms = parseInt(document.getElementById('poll-ms').value, 10);
      const res = await post('/config', { poll_interval_ms: ms });
      toast('nextPollMs set to ' + res.poll_interval_ms + ' ms');
    }

    async function resetAll() {
      await post('/reset');
      toast('Room streams and poll logs cleared');
    }

    // -----------------------------------------------------------------------
    // Contextual prompt button activation & activity rendering
    // -----------------------------------------------------------------------
    function setTriggerActive(id, active, activeClass = 'btn-primary') {
      const btn = document.getElementById(id);
      if (!btn) return;
      btn.className = active ? activeClass : 'btn-waiting';
    }

    function updateSubscriptionStates(activeStreams) {
      const list = activeStreams || [];
      const hbActive = list.some((s) => s.startsWith('system.heartbeat:'));
      const jobStream = list.find((s) => s.startsWith('job.status_changed:'));
      const jobActive = Boolean(jobStream);
      activeJobId = jobStream ? jobStream.slice('job.status_changed:'.length) : null;

      setTriggerActive('trigger-hb', hbActive, 'btn-primary');
      setTriggerActive('trigger-hb-passive', hbActive, 'btn-primary');
      setTriggerActive('trigger-job-run', jobActive, 'btn-primary');
      setTriggerActive('trigger-job-done', jobActive, 'btn-primary');
      setTriggerActive('trigger-gap', jobActive, 'btn-primary');
      setTriggerActive('trigger-revoke', jobActive, 'btn-danger');

      document.getElementById('hint-hb').textContent = hbActive
        ? 'Client is polling system.heartbeat. Click Send heartbeat to wake the agent. Edit the prompt to subscribe once if you want a single trigger.'
        : 'Waiting for a system.heartbeat poll, or click Send heartbeat anytime. Edit the prompt to subscribe once if you want it to stop after the first event.';
      document.getElementById('hint-job').textContent = jobActive
        ? 'Client is polling job.status_changed (' + activeJobId + '). Emit running or done.'
        : 'Waiting for a job.status_changed poll, then emit running and done.';

      const statusEl = document.getElementById('scope-status');
      if (list.length) {
        statusEl.textContent = 'Polling: ' + list.join(', ');
      } else {
        statusEl.textContent = 'Waiting for a client to poll this room';
      }
    }

    function shortTime(iso) {
      if (!iso) return '';
      const d = new Date(iso);
      return Number.isNaN(d.getTime()) ? iso : d.toLocaleTimeString([], { hour12: false });
    }

    function escapeHtml(str) {
      return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    }

    function renderFormatted(data) {
      const evContainer = document.getElementById('events-list');
      const events = (data.recent_events || []).slice().reverse();
      if (!events.length) {
        evContainer.innerHTML = '<p class="empty-state">No events emitted in this room yet.</p>';
      } else {
        evContainer.innerHTML = events.map((ev) => {
          const payload = JSON.stringify(ev.data || {});
          return (
            '<div class="log-item">' +
              '<div class="log-meta">' +
                '<span>' + escapeHtml(ev.event_id) + ' ' + escapeHtml(ev.name) + ' (' + escapeHtml(ev.filter_key) + ')</span>' +
                '<span class="log-time">' + escapeHtml(shortTime(ev.timestamp)) + '</span>' +
              '</div>' +
              '<div class="log-detail">' + escapeHtml(payload) + '</div>' +
            '</div>'
          );
        }).join('');
      }

      const pollContainer = document.getElementById('polls-list');
      const polls = (data.recent_polls || []).slice().reverse();
      if (!polls.length) {
        pollContainer.innerHTML = '<p class="empty-state">Waiting for a client to call events/poll.</p>';
      } else {
        pollContainer.innerHTML = polls.map((p) => {
          const flags = [];
          if (p.returned > 0) flags.push('<span class="status-ok">delivered ' + p.returned + '</span>');
          if (p.truncated) flags.push('<span class="status-warn">truncated</span>');
          if (p.has_more) flags.push('<span class="status-warn">hasMore</span>');
          const cIn = p.cursor_in === null || p.cursor_in === undefined ? 'none' : p.cursor_in;
          const summary = 'cursor ' + escapeHtml(cIn) + ' -> ' + escapeHtml(p.cursor_out) + (flags.length ? '  ' + flags.join('  ') : '');
          return (
            '<div class="log-item">' +
              '<div class="log-meta">' +
                '<span>' + escapeHtml(p.name) + '</span>' +
                '<span class="log-time">' + escapeHtml(shortTime(p.at)) + '</span>' +
              '</div>' +
              '<div class="log-detail">' + summary + '</div>' +
            '</div>'
          );
        }).join('');
      }
    }

    async function refresh() {
      try {
        const r = await fetch(apiPrefix + '/stats');
        const data = await r.json();

        document.getElementById('metric-events').textContent = String(data.event_count || 0);
        document.getElementById('metric-polls').textContent = String(data.poll_count || 0);
        document.getElementById('metric-interval').textContent = String(data.poll_interval_ms) + ' ms';

        updateSubscriptionStates(data.active_streams);
        renderFormatted(data);

        document.getElementById('events-raw').textContent = JSON.stringify({
          room_id: data.room_id,
          poll_interval_ms: data.poll_interval_ms,
          active_streams: data.active_streams,
          streams: data.streams,
          pending_gaps: data.pending_gaps,
          forbidden: data.forbidden,
          recent_events: data.recent_events,
        }, null, 2);
        document.getElementById('polls-raw').textContent = JSON.stringify({
          poll_count: data.poll_count,
          recent_polls: data.recent_polls,
        }, null, 2);
      } catch (err) {
        document.getElementById('polls-raw').textContent = String(err);
      }
    }

    function connectStream() {
      if (typeof EventSource === 'undefined') return;
      const es = new EventSource(apiPrefix + '/stream');
      es.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.kind === 'emit' || msg.kind === 'poll' || msg.kind === 'fault') {
            pushPulse(msg.kind);
            refresh();
          } else if (msg.kind === 'state') {
            refresh();
          }
        } catch {
          refresh();
        }
      };
    }

    setCmdMode('uvx');
    setInterval(refresh, 4000);
    refresh();
    window.addEventListener('load', () => setTimeout(connectStream, 150));
  </script>
</body>
</html>
"""


def _render_dashboard(room_id: str) -> HTMLResponse:
    return HTMLResponse(DASHBOARD_HTML.replace("__ROOM_ID__", room_id))


async def http_root(request: Request) -> RedirectResponse:
    del request
    room_id = registry.new_room_id()
    registry.get_or_create(room_id)
    return RedirectResponse(
        url=f"/r/{room_id}",
        status_code=302,
        headers={"Cache-Control": "no-store"},
    )


async def http_room_index(request: Request) -> Response:
    raw_room = str(request.path_params["room_id"])
    valid = registry.validate_room_id(raw_room)
    if valid is None:
        return RedirectResponse(
            url=f"/r/{registry.new_room_id()}",
            status_code=302,
            headers={"Cache-Control": "no-store"},
        )
    registry.get_or_create(valid)
    return _render_dashboard(valid)


# Disable localhost-only Host header checks so the server works behind reverse
# proxies such as Hugging Face Spaces (*.hf.space).
mcp_app = mcp.streamable_http_app(
    streamable_http_path="/mcp",
    stateless_http=True,
    host="0.0.0.0",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@contextlib.asynccontextmanager
async def lifespan(starlette_app: Starlette):
    del starlette_app
    async with mcp.session_manager.run():
        yield


starlette_app = Starlette(
    lifespan=lifespan,
    routes=[
        Route("/", http_root, methods=["GET"]),
        Route("/new", http_root, methods=["GET"]),
        Route("/r/{room_id}", http_room_index, methods=["GET"]),
        Route("/health", http_health, methods=["GET"]),
        Route("/static/hf-logo.svg", http_static_hf_logo, methods=["GET"]),
        # Room-scoped control plane
        Route("/r/{room_id}/emit", http_emit, methods=["POST"]),
        Route("/r/{room_id}/emit-gap", http_gap, methods=["POST"]),
        Route("/r/{room_id}/terminate", http_terminate, methods=["POST"]),
        Route("/r/{room_id}/unterminate", http_unterminate, methods=["POST"]),
        Route("/r/{room_id}/config", http_config, methods=["POST"]),
        Route("/r/{room_id}/reset", http_reset, methods=["POST"]),
        Route("/r/{room_id}/stats", http_stats, methods=["GET"]),
        Route("/r/{room_id}/stream", http_stream, methods=["GET"]),
        # Public fallback control plane (backwards compatible)
        Route("/emit", http_emit, methods=["POST"]),
        Route("/emit-gap", http_gap, methods=["POST"]),
        Route("/terminate", http_terminate, methods=["POST"]),
        Route("/unterminate", http_unterminate, methods=["POST"]),
        Route("/config", http_config, methods=["POST"]),
        Route("/reset", http_reset, methods=["POST"]),
        Route("/stats", http_stats, methods=["GET"]),
        Route("/stream", http_stream, methods=["GET"]),
        *mcp_app.routes,
    ],
)


class RoomScopeMiddleware:
    """Bind per-room PlaygroundState via ContextVar for /r/<room_id>/mcp."""

    def __init__(self, inner: ASGIApp) -> None:
        self._inner = inner

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._inner(scope, receive, send)
            return

        path: str = scope.get("path", "")
        match = ROOM_MCP_PATH_RE.match(path)
        if match is not None:
            valid = registry.validate_room_id(match.group(1))
            if valid is None:
                response = _invalid_room_response()
                await response(scope, receive, send)
                return
            room_state = registry.get_or_create(valid)
            token = current_room_var.set(room_state)
            rewritten = dict(scope)
            rewritten["path"] = "/mcp"
            if "raw_path" in rewritten:
                rewritten["raw_path"] = b"/mcp"
            try:
                await self._inner(rewritten, receive, send)
            finally:
                current_room_var.reset(token)
            return

        if path in ("/mcp", "/mcp/"):
            token = current_room_var.set(registry.get_or_create(PUBLIC_ROOM_ID))
            try:
                await self._inner(scope, receive, send)
            finally:
                current_room_var.reset(token)
            return

        await self._inner(scope, receive, send)


app: ASGIApp = RoomScopeMiddleware(starlette_app)


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP Events (SEP-3415) Playground Server")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
