# herdr-core integration: handover

Written 16 November 2026 for continuing on a machine with screen capture.
Checked against hc 0.1.18. The companion note for the cache keep-warm work is
`plan/cache-warm-handover.md`.

## Where the code is

- Branch `cache-warm` on `origin` (evalstate/fast-agent).
  `git fetch origin && git switch cache-warm && uv sync`.
- The herdr-core child-attachment work was uncommitted in the checkout and went
  into commit `27fe44327` together with the cache keep-warm feature, because ten
  files held both sets of changes.
- User-facing description: `docs/docs/guides/tui.md`, sections "Agent lifecycle
  status" and "What an idle session is waiting on".
- Instructions the model receives inside hc: `resources/shared/herdr.md`.

## What the integration does today

fast-agent detects herdr-core from `HC_TERMINAL_ID` and `HC_TERMINAL_SOCKET`.
Outside hc none of this runs.

| Behaviour | Where |
| --- | --- |
| Reports lifecycle state (`idle`, `working`, `blocked`, `error`) and a status message over OSC 7501 | `integrations/program_status.py`, `integrations/herdr_lifecycle.py` |
| Reads this terminal's unfinished children from hc and sorts them into "will resume the session" and long-lived services | `integrations/hc_children.py` |
| Prints what the session is still attached to when a turn ends | `ui/interactive_prompt.py`, `ui/prompt/input.py` (`refresh_session_attachments`) |
| Shows the same summary in the bottom toolbar and in the status message | `ui/prompt/status_bar/renderer.py` (`_build_hc_segment`) |
| Holds input that arrives during a turn, so `hc send` text is not echoed into output or left unsubmitted | `ui/prompt/turn_input.py` |
| Treats hc-delivered messages as continuing the current turn, shown under an `hc · …` heading | `ui/message_display_helpers.py`, `history/turns.py` |
| Keeps the prompt cache warm while idle with children that will resume the session | `ui/prompt/cache_warm.py`, `core/cache_warmer.py` |

How children are classified (`read_hc_attachments`):

- It runs `hc wait --children all --timeout 0`, then `hc get <address>` for each
  unfinished child. Each hc call has a 2-second timeout; the snapshot is reused
  for 2 seconds.
- A child labelled `role=service` counts as a service.
- A child counts as "will resume" if hc will announce its exit (`notify`), it
  reports status (`reported`), or it was launched with a first message.
- fast-agent's own `run_in_background` shell processes are counted separately,
  because nothing wakes the session when they end.

The snapshot is refreshed only when a turn ends and when a prompt is created. It
is not refreshed while the prompt sits idle.

## What to look at on screen

Run fast-agent inside an hc terminal (`hc run --bg --name lead fast-agent go …`,
then `hc attach lead`).

Turn-end lines, printed dim after the usage lines:

```text
hc: 2 running - probe-a (t134), probe-b (t135) · will resume when they report or exit · 1 service running
1 background process still running; nothing will resume this session when it ends
```

Bottom toolbar (yellow `hc:` segment, followed by the cache segment):

```text
… | hc: waiting on 1: napper | warm ✓ 2/2 98% next 09:30 until 09:59
… | hc: waiting on 1: napper | cache until 16:41
```

Status message as other terminals see it (`hc get <t>`, field `status.msg`):

```text
waiting on 1: napper · warm ✓ 2/2 98% next 09:30 until 09:59 · 1.89% - gpt-6.1-terra · $0.0005 ($0.0057)
```

Message headings in the transcript:

```text
▎▶ hc · t12x          a peer message sent with hc send
▎▶ hc · t174 exited   hc's own notice that a child ended
```

Suggested captures:

1. Ask the agent to run `hc run --bg --name napper sleep 100` and end its turn.
   Capture the turn-end line, the `hc: waiting on 1: napper` toolbar segment,
   and, about 100 seconds later, the `hc · … exited` notice starting a new
   response.
2. The same with a model on a 5-minute cache (for example
   `copilot.claude-haiku-5.5`) and `CACHE_WARM__INTERVAL_SECONDS=20`: `hc`
   keep-warm lines appear in scrollback while the session waits.
3. Start a child with `--label role=service`; it is counted as a service and
   not listed by name.
4. Start a `run_in_background` shell process and end the turn; capture the
   "nothing will resume this session" line.
5. From a second terminal, `hc send lead "…"` while the lead is mid-turn; the
   message should be held and consumed at the next tool-loop boundary, not
   echoed into the streaming output.
6. From a second terminal, `hc get lead` before, during and after a turn to see
   the state and status message change.

## What was tested in this session

- Child started by the model with `hc run --bg --name napper sleep 100`: the
  turn-end line and toolbar segment appeared, keep-warm requests ran with
  reason `hc`, and the exit notice woke the session.
- `hc send` to a running session, used throughout for driving test sessions.
- The status message carried the hc summary, the cache segment, context and
  cost together.

Observed once and not investigated: after being told to end its turn without
waiting, the model returned an empty completion twice when the "napper exited"
notice arrived, and fast-agent reported "Completion returned no assistant
content or tool calls after one retry". The session recovered on the next
message. This looks like the model having nothing to say, but that is unconfirmed.

## Limits of the current integration

Verified from hc's CLI help and API schema; no resume experiment was run.

- **fast-agent learns about children only by observation.** The model types
  `hc run --bg …` as shell text. Nothing ties that command to the child, and the
  child is not recorded in the fast-agent session.
- **Ownership is by terminal, not by session.** A child's parent is the hc
  terminal. Children end with the parent's program unless started with
  `--outlive-parent`.
- **Resumption.** Restarting fast-agent in the same terminal with
  `--outlive-parent` children should find them again (untested). Resuming in a
  different terminal does not: the API sets a parent only at creation, and wake-up
  notes still go to the old terminal.
- **Cost.** hc has no usage or cost fields. The only carrier is the free-text
  status message, where a fast-agent child writes something like
  `$0.0053 ($0.0053)`. Child cost cannot be added to the parent reliably.
- **Staleness.** The toolbar's `hc:` segment is refreshed at turn and prompt
  boundaries only. A child that exits without a notice keeps showing as running
  until the next prompt.
- **No MCP server.** hc exposes an OpenRPC API (`hc schema`, served over
  stdin/stdout by `hc connect`) with about 30 methods such as `terminal.run`,
  `terminal.send`, `terminal.wait` and `events.subscribe`. `herdr-app` was not
  installed on the original machine, so whether the app adds anything is
  unchecked.
- **Codex coordinators.** Over the Codex websocket the cache cannot be warmed
  while a turn is blocked on tools. A coordinator that ends its turn and waits
  idle is covered; one that waits inside a turn is not.

Compared with a managed background process:

| | Managed background process | hc child |
| --- | --- | --- |
| Who owns it | fast-agent | hc core |
| How fast-agent knows about it | it launched it: id, status, output | polls hc at prompt boundaries |
| Survives a fast-agent restart | only if launched as durable | only with `--outlive-parent` |
| Wakes the session when done | no | yes, by typing a note |
| Recorded in the session | yes | no |
| Usage and cost | not applicable | not available |
| A person can attach and steer | no | yes |

## Opportunities, smallest first

None of this is built, and no decision has been made.

Small, and fixes visible gaps:

1. **Record children in the session.** Note each child's address and label it
   with the session id, so a resumed session can find it with
   `hc ls --label`. Prerequisite for most of what follows.
2. **Use hc's event stream** (`hc events`, `events.subscribe`) so the toolbar
   updates live and silently exited children stop showing as running.
3. **Say what happened on wake-up**: which child reported and how long it ran,
   not only the raw `<hc>` message.

Medium, and makes children feel like subagents:

4. **Show children in the progress display**, like the subagent table, using
   hc's per-terminal state and status message. This would replace many
   `hc read` calls by the model.
5. **Roll up cost** for fast-agent children by passing a session id the parent
   can read back. Other harnesses would stay unavailable.
6. **Surface `blocked` children** in the parent's toolbar and status, instead of
   waiting for hc's note.

Larger, and changes how the model drives hc:

7. **A thin first-class tool** (spawn, send, wait, list) over hc's API.
   fast-agent would then know what it started rather than inferring it, which
   makes items 1 to 6 reliable. The cost is a second surface to keep in step
   with a young, fast-moving CLI that already teaches itself through
   `hc --skill`; the tool must appear only when hc is present, and adding it
   changes the tool list and so the cached prefix.
8. **An MCP adapter over hc's API.** Similar benefit and usable by other
   harnesses, but ours to write and maintain.
9. **A child-wait tool that returns to the model on a timer**, folded like
   process polls. This is the route to keeping a Codex coordinator's cache warm
   while it waits inside a turn.

Suggested starting point: items 1 and 2. Both are small, they fix the two things
that are visibly wrong today (children lost on resume, stale toolbar), and they
do not commit to a tool or MCP design. The tool question can wait until it is
clear whether models keep misusing the CLI.

## Useful commands

```bash
hc --skill                      # the manual the model reads
hc ls                           # all live terminals
hc get <t>                      # one terminal: state, status message, labels, parent
hc wait --children all --timeout 0   # what fast-agent asks to find its children
hc events                       # live change stream
hc schema                       # the API as OpenRPC

# Tests for the integration
uv run pytest tests/unit/fast_agent/integrations -q
uv run pytest tests/unit/fast_agent/ui/test_turn_input.py \
  tests/unit/fast_agent/agents/test_hc_user_message_display.py -q
```
