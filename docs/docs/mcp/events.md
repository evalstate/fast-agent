---
title: MCP Events
social:
  title: MCP Events
  tagline: Subscribe agents to events from MCP Servers.
  description: Subscribe agents to events from MCP Servers and wake them when one arrives.
  alt: fast-agent social card - MCP Events
---

## Compatibility

`fast-agent` implements the client side of the
[Events Extension](https://modelcontextprotocol.io/seps/3415-events-extension)
(SEP-3415, `io.modelcontextprotocol/events`) with poll and push delivery.

It uses `events/list`, `events/poll` and `events/stream`. Webhook delivery is
not implemented, so an event type that offers only webhook is listed but cannot
be subscribed to. The extension is advertised to every server, and the
commands below work for any server that declares it in return. The servers this
has been run against declare it on the 2026-07-28 protocol, so set
`protocol_mode: modern` if `auto` does not negotiate it.

## Subscribing

A subscription names a server, an event type and the arguments that event type
takes. Use `/events` in an interactive session:

```text
/events list
/events subscribe jobs job.status_changed job_id=abc123
/events subscribe jobs job.status_changed job_id=abc123 --poll
/events
/events poll <id>
/events unsubscribe <id>
/events unsubscribe all
```

`/events list [server]` shows what each attached server offers, with the
delivery modes and argument names. An event type whose `x-mcp-header`
annotations are invalid is left out. Arguments are given as `key=value` pairs,
or as one JSON object. Values that parse as JSON keep their type. `/events`
with no argument lists subscriptions with their status and the number of events
delivered. The id shown there is derived from the server, event name and
arguments, so subscribing twice to the same thing gives one subscription.

A subscription is delivered by push when the event type offers it, and by poll
otherwise. `--poll` or `--push` picks the mode. In both modes `fast-agent` drops
events whose `eventId` it has already seen.

A push subscription holds one `events/stream` request open, and the server
sends events on it as they occur. The server also sends a heartbeat when it has
nothing else to send. When no message arrives for three heartbeat intervals, or
the stream drops or the server closes it, `fast-agent` opens a new stream from
the last cursor after one to two seconds.

A poll subscription runs one poll loop. It follows the server's `nextPollMs`
with a floor of one second and polls again at once while the server reports
`hasMore`.

A poll that fails, or a stream that cannot be opened, is retried with a doubling
delay of up to five minutes, from the same cursor. `/events poll <id>` ends that
wait. When a poll
or a stream is rejected with a header mismatch, the event types are listed again
and the request is tried once more from the same cursor. A second mismatch ends
the subscription.

## How events reach the model

Delivered events wait in an inbox and the prompt toolbar shows a count. Before
each model call the inbox is drained into one user message, after the user's own
message on the first step of a turn. Events that arrive while a turn is running
join that turn at its next model call.

The inbox holds 256 items. When it is full the oldest item is dropped, and the
model is told that events from that subscription were skipped.

!!! warning "Event payloads are untrusted"

    The message tells the model that the payloads are data from an external
    system and not instructions, and wraps each one in an `<mcp-event>` block.
    That is a label and not a guarantee. A server you subscribe to can put text
    in front of the model, and with `--wake` it can do so while nobody is
    watching. Subscribe only to servers you trust.

If the server reports that events were skipped, or ends the subscription, the
model is told in the same message. Read the current state with tools when that
matters.

## Waking an idle agent

Interactive subscriptions (`/events subscribe`, `/events select`, and the
`events` tool) wake an idle agent by default. Pass `--no-wake` (or `wake=False`)
to buffer events passively until the next turn instead:

```text
/events subscribe jobs job.status_changed job_id=abc123 --no-wake
```

Subscriptions declared in `fastagent.config.yaml` remain passive (`wake: false`)
by default unless `wake: true` is set.

While the prompt is idle, an event from a subscription with wake set ends the
prompt and starts a turn whose input is the event message. Text you were typing
is put back in the next prompt. `Ctrl+C` cancels the turn as usual. Only the
agent shown in the prompt is woken; subscriptions belong to the agent that made
them. Subscribing again with the same arguments updates the flag.

The agent's instruction should say what to do when events arrive, since the
wake-up turn carries no other request.

## Declaring subscriptions

Subscriptions can be declared on a server entry so they start for each agent
that uses that server:

```yaml
mcp:
  servers:
    jobs:
      url: http://127.0.0.1:8000/mcp
      protocol_mode: modern
      events:
        - name: job.status_changed
          arguments:
            job_id: abc123
          wake: true
```

The same `events` list is accepted on an AgentCard `mcp_connect` entry, and on
`MCPServerSettings` passed to `attach_mcp_server` at runtime. An entry can set
`delivery: poll` or `delivery: push` to pick the mode. Declared subscriptions
start when the agent is first used: at its first turn, or when its prompt first
waits for input. An agent that is loaded and never used does not subscribe.
Every agent that uses the server gets its own subscriptions. A declared event
type the server does not offer is logged as a warning and skipped. Detaching a
server ends its subscriptions, declared or not.

## From code

The subscription manager is available on any MCP agent. A host that embeds
`fast-agent` can subscribe, wait and send without the TUI:

```python
async with fast.run() as app:
    agent = app.seat
    manager = agent.aggregator.events
    await manager.subscribe("jobs", "job.status_changed", {"job_id": "abc123"}, wake=True)

    while True:
        await agent.wait_for_event_wake()
        await agent.send("Events arrived. Act on them, then end your turn.")
```

`wait_for_event_wake()` returns while an event from a subscription with wake set
is still in the inbox, and the turn drains it. It also starts the declared
subscriptions, as `agent.start_configured_events()` does for a host that needs
them before its first turn. `subscribe` takes `delivery="poll"` or
`delivery="push"` to pick the mode. `manager.poll_now(key)` polls a
subscription immediately and keeps its cursor. A provider failure comes back
from `send` as a reply on the error channel after the events have been drained,
so a host that must not lose them should check for that and retry.
`examples/mcp/events/event_loop.py` is a complete driver with retries, token
usage and transcript export.

## Current scope

- Poll and push delivery. Webhook delivery is not implemented.
- `Mcp-Param-*` headers are not sent for arguments annotated with
  `x-mcp-header`. A server that checks them rejects the poll or the stream.
- `/events` is available in the interactive TUI, not over ACP.
- Subscriptions made with `/events` last for the session. Declared
  subscriptions start again from "now" on the next run.
- An idle agent under `fast-agent serve` is not woken.
