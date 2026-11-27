"""Standalone Streamable HTTP MCP Events (SEP-3415) playground server.

Provides a poll-based SEP-3415 MCP server at ``/mcp`` alongside a browser
control plane at ``/`` for triggering events, testing truncated gaps,
subscription revocations, ``maxAgeMs``, ``maxEvents``, and ``--resume``.

Run locally:
    uv run examples/mcp/events/playground/app.py --port 8765

Deploy to Hugging Face Spaces (Docker Space, port 7860):
    Upload the contents of this directory to a Docker Space repository.
"""

import argparse
import contextlib
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

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
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse
from starlette.routing import Route

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
# In-memory playground state
# ---------------------------------------------------------------------------

MAX_STREAM_HISTORY = 500
MAX_POLL_LOG = 200
MAX_RECENT_EVENTS = 100


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
    poll_interval_ms: int = field(
        default_factory=lambda: max(250, int(os.environ.get("POLL_INTERVAL_MS", "2000")))
    )
    streams: dict[tuple[str, str], list[StoredEvent]] = field(default_factory=dict)
    stream_totals: dict[tuple[str, str], int] = field(default_factory=dict)
    pending_gaps: set[tuple[str, str]] = field(default_factory=set)
    forbidden_keys: set[tuple[str, str]] = field(default_factory=set)
    poll_log: list[dict[str, Any]] = field(default_factory=list)
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    event_seq: int = 0

    def emit(self, event_name: str, filter_key: str, data: dict[str, Any]) -> StoredEvent:
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
        return record

    def reset(self) -> None:
        # Keep stream_totals and event_seq monotonic so connected clients see
        # truncated=True and fresh event IDs instead of silently skipping events.
        self.streams.clear()
        self.pending_gaps.clear()
        self.forbidden_keys.clear()
        self.poll_log.clear()
        self.recent_events.clear()


state = PlaygroundState()


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
    key = _resolve_stream_key(params.name, params.arguments)

    if key in state.forbidden_keys:
        raise MCPError(EVENTS_FORBIDDEN, "Subscription revoked by server")

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
    state.emit(
        "job.status_changed",
        job_id,
        {"job_id": job_id, "stage": stage, "detail": detail},
    )
    return f"Job {job_id} -> {stage}"


# ---------------------------------------------------------------------------
# HTTP control plane & browser UI
# ---------------------------------------------------------------------------


async def http_health(request: Request) -> JSONResponse:
    del request
    return JSONResponse({"status": "ok"})


async def http_emit(request: Request) -> JSONResponse:
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
    body = await request.json()
    key = (
        str(body.get("event_name") or "job.status_changed"),
        str(body.get("filter_key") or body.get("job_id") or "*"),
    )
    state.pending_gaps.add(key)
    return JSONResponse({"gap_armed": list(key)})


async def http_terminate(request: Request) -> JSONResponse:
    body = await request.json()
    key = (
        str(body.get("event_name") or "job.status_changed"),
        str(body.get("filter_key") or body.get("job_id") or "*"),
    )
    state.forbidden_keys.add(key)
    return JSONResponse({"forbidden": list(key)})


async def http_unterminate(request: Request) -> JSONResponse:
    body = await request.json()
    key = (
        str(body.get("event_name") or "job.status_changed"),
        str(body.get("filter_key") or body.get("job_id") or "*"),
    )
    state.forbidden_keys.discard(key)
    return JSONResponse({"restored": list(key)})


async def http_config(request: Request) -> JSONResponse:
    body = await request.json()
    if "poll_interval_ms" in body:
        try:
            state.poll_interval_ms = max(250, min(int(body["poll_interval_ms"]), 300_000))
        except (TypeError, ValueError):
            return JSONResponse({"error": "poll_interval_ms must be an integer"}, status_code=400)
    return JSONResponse({"poll_interval_ms": state.poll_interval_ms})


async def http_reset(request: Request) -> JSONResponse:
    del request
    state.reset()
    return JSONResponse({"reset": True})


async def http_stats(request: Request) -> JSONResponse:
    del request
    return JSONResponse(
        {
            "poll_interval_ms": state.poll_interval_ms,
            "streams": {f"{k[0]}:{k[1]}": len(v) for k, v in state.streams.items()},
            "pending_gaps": [f"{k[0]}:{k[1]}" for k in sorted(state.pending_gaps)],
            "forbidden": [f"{k[0]}:{k[1]}" for k in sorted(state.forbidden_keys)],
            "poll_count": len(state.poll_log),
            "recent_events": state.recent_events[-15:],
            "recent_polls": state.poll_log[-15:],
        }
    )


DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MCP Events Playground (SEP-3415)</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #0f1115;
      --panel: #171a21;
      --border: #2a2f3a;
      --text: #e6e9ef;
      --muted: #9aa3b2;
      --accent: #6ea8fe;
      --danger: #f06a6a;
      --ok: #5cd68d;
    }
    * { box-sizing: border-box; }
    body {
      font-family: system-ui, -apple-system, Segoe UI, sans-serif;
      max-width: 920px;
      margin: 1.5rem auto;
      padding: 0 1rem 3rem;
      background: var(--bg);
      color: var(--text);
      line-height: 1.45;
    }
    h1 { font-size: 1.4rem; margin: 0 0 .35rem; }
    h2 { font-size: 1rem; margin: 1.25rem 0 .5rem; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem;
      margin: .75rem 0;
    }
    .row { display: flex; flex-wrap: wrap; gap: .5rem; align-items: center; margin: .5rem 0; }
    label { font-size: .88rem; color: var(--muted); display: inline-flex; align-items: center; gap: .4rem; }
    input {
      padding: .42rem .6rem;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: var(--bg);
      color: var(--text);
      font-size: .9rem;
    }
    button {
      padding: .46rem .85rem;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: #222733;
      color: var(--text);
      cursor: pointer;
      font-size: .88rem;
    }
    button:hover { background: #2c3342; border-color: #3d4659; }
    button.primary { border-color: #3a5c96; background: #1c2e4a; }
    button.primary:hover { background: #243b5e; }
    button.danger { border-color: #6e2d2d; background: #361919; color: #ffb3b3; }
    button.danger:hover { background: #4a2020; }
    code {
      background: var(--bg);
      border: 1px solid var(--border);
      padding: .18rem .42rem;
      border-radius: 5px;
      font-size: .86rem;
      word-break: break-all;
    }
    pre {
      background: var(--bg);
      border: 1px solid var(--border);
      padding: .75rem;
      border-radius: 8px;
      overflow: auto;
      font-size: .82rem;
      max-height: 360px;
      margin: .4rem 0 0;
    }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: .75rem; }
    @media (max-width: 720px) { .grid { grid-template-columns: 1fr; } }
    .toast { font-size: .84rem; color: var(--ok); min-height: 1.2rem; }
  </style>
</head>
<body>
  <h1>MCP Events Playground (SEP-3415)</h1>
  <div class="card">
    <div class="row">
      <span>Endpoint:</span>
      <code id="mcp-url"></code>
      <button onclick="copyText(mcpUrl)">Copy URL</button>
      <button class="primary" onclick="copyText('fast-agent go --url ' + mcpUrl)">Copy fast-agent go</button>
      <button onclick="copyText('/mcp connect ' + mcpUrl + ' --name playground')">Copy /mcp connect</button>
    </div>
    <div class="toast" id="toast"></div>
  </div>

  <div class="card">
    <div class="row">
      <label>job_id <input id="job" value="job-1" size="14"></label>
      <label>detail <input id="detail" value="manual trigger" size="24"></label>
      <label>nextPollMs <input id="poll-ms" type="number" value="2000" min="250" max="300000" step="250" style="width:95px"></label>
      <button onclick="setPollMs()">Apply poll interval</button>
    </div>
    <div class="row">
      <button class="primary" onclick="emitJob('queued')">Emit queued</button>
      <button class="primary" onclick="emitJob('running')">Emit running</button>
      <button class="primary" onclick="emitJob('done')">Emit done</button>
      <button class="danger" onclick="emitJob('failed')">Emit failed</button>
      <button onclick="emitHB()">Emit system.heartbeat</button>
    </div>
    <div class="row">
      <button onclick="armGap()">Arm truncated gap (job)</button>
      <button class="danger" onclick="armTerm()">Revoke / Forbidden (job)</button>
      <button onclick="unTerm()">Unrevoke (job)</button>
      <button class="danger" onclick="resetAll()">Reset all state</button>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Streams &amp; Recent Events</h2>
      <pre id="events">Loading...</pre>
    </div>
    <div class="card">
      <h2>Recent Polls</h2>
      <pre id="polls">Loading...</pre>
    </div>
  </div>

  <script>
    const mcpUrl = window.location.origin + '/mcp';
    document.getElementById('mcp-url').textContent = mcpUrl;

    function toast(msg) {
      const el = document.getElementById('toast');
      el.textContent = msg;
      setTimeout(() => { if (el.textContent === msg) el.textContent = ''; }, 2500);
    }

    async function copyText(text) {
      try {
        await navigator.clipboard.writeText(text);
        toast('Copied: ' + text);
      } catch {
        toast(text);
      }
    }

    async function post(path, body = {}) {
      const r = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await r.json();
      await refresh();
      return data;
    }

    function jobId() { return document.getElementById('job').value.trim() || 'job-1'; }
    function detail() { return document.getElementById('detail').value; }

    async function emitJob(stage) {
      const res = await post('/emit', {
        event_name: 'job.status_changed',
        job_id: jobId(),
        stage,
        detail: detail(),
      });
      toast('Emitted ' + res.event_id + ' (' + stage + ')');
    }

    async function emitHB() {
      const res = await post('/emit', {
        event_name: 'system.heartbeat',
        filter_key: '*',
        data: { seq: Date.now(), note: detail() },
      });
      toast('Emitted ' + res.event_id + ' (system.heartbeat)');
    }

    async function armGap() {
      await post('/emit-gap', { event_name: 'job.status_changed', job_id: jobId() });
      toast('Armed truncated gap for ' + jobId());
    }

    async function armTerm() {
      await post('/terminate', { event_name: 'job.status_changed', job_id: jobId() });
      toast('Revoked subscription for ' + jobId());
    }

    async function unTerm() {
      await post('/unterminate', { event_name: 'job.status_changed', job_id: jobId() });
      toast('Unrevoked subscription for ' + jobId());
    }

    async function setPollMs() {
      const ms = parseInt(document.getElementById('poll-ms').value, 10);
      const res = await post('/config', { poll_interval_ms: ms });
      toast('nextPollMs set to ' + res.poll_interval_ms + ' ms');
    }

    async function resetAll() {
      await post('/reset');
      toast('All streams and poll logs cleared');
    }

    async function refresh() {
      try {
        const r = await fetch('/stats');
        const data = await r.json();
        document.getElementById('events').textContent = JSON.stringify({
          poll_interval_ms: data.poll_interval_ms,
          streams: data.streams,
          pending_gaps: data.pending_gaps,
          forbidden: data.forbidden,
          recent_events: data.recent_events,
        }, null, 2);
        document.getElementById('polls').textContent = JSON.stringify({
          poll_count: data.poll_count,
          recent_polls: data.recent_polls,
        }, null, 2);
      } catch (err) {
        document.getElementById('polls').textContent = String(err);
      }
    }

    setInterval(refresh, 1500);
    refresh();
  </script>
</body>
</html>
"""


async def http_index(request: Request) -> HTMLResponse:
    del request
    return HTMLResponse(DASHBOARD_HTML)


# Disable localhost-only Host header checks so the server works behind reverse
# proxies such as Hugging Face Spaces (*.hf.space).
mcp_app = mcp.streamable_http_app(
    streamable_http_path="/mcp",
    stateless_http=True,
    host="0.0.0.0",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@contextlib.asynccontextmanager
async def lifespan(app: Starlette):
    del app
    async with mcp.session_manager.run():
        yield


app = Starlette(
    lifespan=lifespan,
    routes=[
        Route("/", http_index, methods=["GET"]),
        Route("/health", http_health, methods=["GET"]),
        Route("/emit", http_emit, methods=["POST"]),
        Route("/emit-gap", http_gap, methods=["POST"]),
        Route("/terminate", http_terminate, methods=["POST"]),
        Route("/unterminate", http_unterminate, methods=["POST"]),
        Route("/config", http_config, methods=["POST"]),
        Route("/reset", http_reset, methods=["POST"]),
        Route("/stats", http_stats, methods=["GET"]),
        *mcp_app.routes,
    ],
)


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP Events (SEP-3415) Playground Server")
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
