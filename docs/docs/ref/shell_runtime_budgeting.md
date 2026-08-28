---
title: Shell Runtime Budgeting
description: Understand foreground auto-await, process waits, and outer run budgets.
social:
  title: Shell Runtime Budgeting
  tagline: Keep long-running shell work interruptible and budget-aware.
  description: Understand foreground auto-await, process waits, and outer run budgets.
  alt: fast-agent social card — Shell Runtime Budgeting
---

# Foreground auto-await and outer-budget-aware process waits

Status: engineering behavior note and design proposal.

This document separates two related shell-runtime behaviors that affect
long-horizon coding agents:

1. foreground commands now remain inside their original tool call for up to
   `foreground_auto_await_max_seconds`; and
2. later `process(wait)` calls are bounded by a static per-call maximum, but
   not by the remaining outer agent/run budget.

The first behavior is an intentional Fast-Agent source change. The second is
not a newly introduced wait ceiling: waits up to 3,600 seconds were already
supported. It is a newly exposed interaction between longer foreground
auto-await, model-selected waits, and an external harness timeout that
Fast-Agent does not currently know about.

The goals of this note are to:

- make the model-control tradeoff explicit;
- define the difference between shell and outer-run budgets;
- describe the observed long-tail failure mode;
- propose budget-aware semantics that preserve quick-command efficiency; and
- provide concrete unit and integration acceptance criteria.

## Terminology

Several independent timers are easy to conflate.

### Initial foreground yield checks

The shell runtime has initial checks for no-output/idle and total foreground
runtime. For the observed Luna profile these caused live foreground work to
become model-visible after approximately ten seconds before foreground
auto-await was introduced.

### Foreground auto-await cap

`shell_execution.foreground_auto_await_max_seconds` is measured from process
start. After an initial yield condition, Fast-Agent may keep awaiting the
same process inside the original shell tool invocation until this total
runtime cap.

Current default:

```yaml
shell_execution:
  foreground_auto_await_max_seconds: 30
```

Reaching the cap returns a live managed-process ID. It does not terminate the
process.

### Shell hard timeout

An explicit shell `timeout`/`timeout_seconds` is destructive: reaching it
terminates the command. It is distinct from yielding or waiting for a live
process.

### Managed-process wait

`process(action="wait", wait_sec=N)` waits for a managed process but does not
terminate it merely because the wait interval expires.

Current static ceiling:

```yaml
shell_execution:
  process_poll_max_wait_seconds: 3600
```

### Outer agent/run timeout

An embedding harness may terminate the whole Fast-Agent process after a task
budget, for example 5,400 seconds. In external-harness operation this
deadline may be enforced outside Fast-Agent, so the shell runtime has no
remaining-budget signal.

This is the critical missing input.

## Behavior 1: foreground auto-await changes model interruptibility

Source change:

```text
commit 3a80724ee36b27652f22ffc812ea40dfe28c9fbd
feat(shell): auto-await finite foreground commands
```

The feature keeps finite foreground work in its original shell call. This
avoids a model turn whose only purpose is to wait for a command that would
have completed shortly afterward.

### Previous behavior

A representative long foreground command behaved approximately as follows:

```text
t=0s       command starts
t≈10s      shell tool returns a live process
            model can inspect, wait, stop, or do other work
t≈10s+     model chooses the next process action
```

### Prior 240-second default behavior

With a 240-second auto-await cap:

```text
t=0s       command starts
t≈10s      initial foreground yield condition occurs internally
t≈10–240s  Fast-Agent continues waiting inside the original tool call
t<240s     if command finishes, final result returns directly
t=240s     otherwise a live process is returned to the model
```

The current 30-second default returns control at the existing total foreground
yield boundary while still allowing quiet commands that reach the initial
approximately ten-second yield to finish in their original tool call.

During the auto-await interval:

- the model cannot inspect intermediate process state;
- the model cannot stop a hung command;
- the model cannot change strategy;
- no model turn is spent polling; and
- the outer harness budget continues to elapse.

### Intended benefit

This optimization is valuable when:

- commands usually finish within the cap;
- a model would otherwise issue a low-information wait turn;
- provider calls are expensive or slow;
- repeated process polling bloats history; or
- reducing cache-expiry risk is more important than early control.

### Long-tail cost

The same behavior is harmful when:

- tests hang;
- a command is unexpectedly running the full suite;
- output reveals a failure that should trigger early termination;
- a server remains in the foreground;
- the task has a short outer deadline; or
- the model needs time after testing to inspect, repair, commit, and submit.

The relevant design question is not simply “does auto-await save turns?” It
is:

> How much model interruptibility can be traded for fewer scheduling turns
> given the remaining task budget and command risk?

### DeepSWE evidence

In paired Luna/max runs:

| Metric | Fast-Agent 0.10.4 | Fast-Agent 0.10.11 run 1 | Fast-Agent 0.10.11 run 2 |
|---|---:|---:|---:|
| Foreground calls returned running | 437 | 29 | 33 |
| Median time at return/yield | 10.05 s | 240.00 s | 240.00 s |
| Process wait/status actions | 490 | 49 | 40 |
| Trials using process tool | 98 | 27 | 32 |

The feature clearly achieved its immediate objective: model-issued process
polling fell by roughly 90%.

However, mean task duration increased about 28–31%, and nine outer agent
timeouts appeared across two current runs versus zero in the reference run.
Concurrency and provider conditions differ, so those aggregate changes are
not proof of package-only causality. Individual traces do establish the
long-wait mechanism.

## Behavior 2: managed-process waits ignore the outer run budget

Fast-Agent validates a process wait against a static range and configured
maximum. It does not currently clamp the wait against the remaining external
agent deadline.

For example, the model may request:

```json
{
  "process_id": "process-8",
  "action": "wait",
  "wait_sec": 3600
}
```

That request is valid under the current 3,600-second per-call ceiling even
when:

- the task has a 5,400-second outer limit;
- 1,000 seconds have already elapsed;
- the process already consumed 240 seconds in foreground auto-await; and
- the agent still needs time to diagnose, edit, retest, commit, and submit.

### Important source-attribution distinction

The 3,600-second wait ceiling is not itself a new 0.10.11 source default.
What changed operationally is:

1. foreground work remains hidden inside `exec` much longer;
2. when it finally yields, Luna more frequently selects a very long wait;
3. Fast-Agent accepts the wait because it lacks the external deadline; and
4. the harness may cancel the entire process before finalization.

This is an interaction bug or missing contract, not simply a bad constant.

### Direct trace example

On `mnamer-daemon-watch-lifecycle`, the current runtime followed this shape:

```text
pytest -q
  -> approximately 240 seconds foreground auto-await

process(wait, wait_sec=3600)
  -> process still running after approximately 3,843 seconds total

process(stop)
  -> model regains control, but most of the task budget is gone

later
  -> outer harness cancels at 5,400 seconds
```

The older runtime on the same task yielded after approximately ten seconds,
used shorter wait/status cycles, stopped the hung process, continued, and
passed.

## Proposed budget contract

Fast-Agent needs an optional run-budget signal from its caller.

The shell runtime should not assume that every invocation has an outer
deadline, and it should preserve existing behavior when no deadline is
provided.

### Suggested input

At Fast-Agent process startup, the embedding adapter may provide either:

```text
run_budget_seconds
```

or an absolute wall-clock deadline:

```text
run_deadline_epoch_seconds
```

Fast-Agent should convert this once to a local monotonic deadline.

An absolute monotonic value must not be passed across process/container
boundaries because monotonic clocks are host-local.

Possible integration surfaces:

- CLI option;
- harness adapter kwarg;
- environment variable set by the adapter;
- programmatic run context; or
- an optional `RunBudget` object injected into agent and tool runtimes.

### Suggested runtime object

Conceptually:

```python
@dataclass(frozen=True)
class RunBudget:
    deadline_monotonic: float
    finalization_reserve_seconds: float = 600

    def remaining_seconds(self) -> float: ...
    def usable_tool_seconds(self) -> float:
        return max(
            self.remaining_seconds() - self.finalization_reserve_seconds,
            0,
        )
```

The object should be optional. Absence means no known outer budget.

## Proposed behavior 1 improvement: adaptive auto-await

Instead of treating 240 seconds as unconditional, calculate:

```text
effective_auto_await =
    min(
        configured_auto_await_cap,
        remaining_outer_budget - finalization_reserve
    )
```

Then apply additional policy based on command risk.

### Conservative first implementation

The minimum safe change is budget clamping:

```python
effective_cap = configured_cap
if run_budget is not None:
    effective_cap = min(effective_cap, run_budget.usable_tool_seconds())
```

If `effective_cap <= initial_yield_elapsed`, return the live process
immediately at the initial yield boundary.

### More adaptive implementation

An optional policy may reduce the cap for commands likely to hang:

- full repository tests;
- watch/dev/server commands;
- daemons;
- commands with no recent output;
- commands that have crossed historical duration percentiles; or
- commands started late in the run.

Risk classification must remain advisory. Command-name heuristics are
imperfect and should not silently terminate work.

### Preserve process semantics

Clamping auto-await must:

- return a live process rather than kill it;
- preserve all output;
- preserve the process ID;
- preserve explicit background behavior;
- preserve explicit hard-timeout behavior; and
- report why the effective cap was shortened.

## Proposed behavior 2 improvement: sliced, budget-aware waits

Calculate:

```text
effective_wait =
    min(
        requested_wait,
        process_poll_max_wait_seconds,
        process_wait_slice_seconds,
        remaining_outer_budget - finalization_reserve
    )
```

Suggested new setting:

```yaml
shell_execution:
  process_wait_slice_seconds: 240
  finalization_reserve_seconds: 600
```

The slice limit ensures the model periodically regains control even when it
requests 3,600 seconds.

If no outer budget is known, the first implementation may retain the existing
static behavior for compatibility or still apply the wait slice as a separate
policy.

### Near finalization reserve

When:

```text
remaining_outer_budget <= finalization_reserve
```

a process wait should return immediately with the process still running and
structured guidance such as:

```text
Wait not started: the run has entered its reserved finalization window.
Inspect available output, stop unnecessary processes, commit, and finish.
```

Fast-Agent should not automatically commit. The reserve restores model
control so task-specific finalization can occur.

### Wait expiration

An effective wait slice expiring should not be reported as a command failure.
It should return:

```json
{
  "status": "running",
  "requested_wait_seconds": 3600,
  "effective_wait_seconds": 240,
  "wait_clamped": true,
  "clamp_reasons": [
    "process_wait_slice"
  ]
}
```

If budget also clamps the wait:

```json
{
  "remaining_run_seconds": 710,
  "finalization_reserve_seconds": 600,
  "effective_wait_seconds": 110,
  "clamp_reasons": [
    "remaining_run_budget"
  ]
}
```

## Telemetry requirements

Result/session telemetry should expose cumulative and per-call values.

### Per shell invocation

```text
foreground_initial_yield_seconds
configured_auto_await_cap_seconds
effective_auto_await_cap_seconds
foreground_auto_await_elapsed_seconds
auto_await_outcome
auto_await_clamp_reasons
remaining_run_seconds_before
remaining_run_seconds_after
```

### Per process wait

```text
requested_wait_seconds
effective_wait_seconds
actual_wait_seconds
wait_clamped
wait_clamp_reasons
process_status_after_wait
remaining_run_seconds_before
remaining_run_seconds_after
```

### Run totals

```text
total_foreground_auto_await_seconds
auto_await_cap_crossings
budget_clamped_auto_await_count
total_process_wait_seconds
budget_clamped_process_wait_count
wait_slice_count
time_entered_finalization_reserve
```

Without these fields, benchmark analysis must reconstruct behavior from tool
results and timestamps.

## Configuration proposal

One possible backward-compatible shape:

```yaml
shell_execution:
  foreground_auto_await_max_seconds: 30
  foreground_auto_await_budget_aware: true
  process_poll_max_wait_seconds: 3600
  process_wait_slice_seconds: 240
  finalization_reserve_seconds: 600
```

Questions to resolve:

1. Should `process_wait_slice_seconds` default to the model catalogue polling
   cadence or a global value?
2. Should the finalization reserve be a shell setting, run setting, or harness
   setting?
3. Should no-budget standalone sessions preserve unlimited long waits?
4. Should explicit command hard timeouts also be clamped to the run budget?
5. Should active output extend auto-await, or should total runtime remain the
   only cap?
6. Should the model receive a remaining-budget field on every tool result or
   only when clamping occurs?

## Acceptance tests

### Foreground auto-await unit tests

1. **Quick command**
   - configured cap: 240 seconds;
   - command finishes in 2 seconds;
   - result returns normally in the original tool call.

2. **Cap crossing**
   - configured cap: 240 seconds;
   - no outer budget;
   - long command returns a live process at approximately 240 seconds;
   - process is not terminated.

3. **Disabled behavior**
   - configured cap: 0;
   - long command returns at the initial yield boundary;
   - prior process semantics are preserved.

4. **Budget clamp**
   - configured cap: 240 seconds;
   - usable outer budget: 60 seconds;
   - live process returns no later than approximately 60 seconds;
   - metadata records `remaining_run_budget`.

5. **Already in reserve**
   - usable outer budget: 0;
   - live process returns at the initial yield boundary;
   - no extra auto-await occurs.

6. **Output preservation**
   - command emits output before and after initial yield;
   - output is complete and ordered after cap crossing.

### Process-wait unit tests

1. **Requested wait below all limits**
   - request: 30 seconds;
   - effective wait: 30 seconds.

2. **Slice clamp**
   - request: 3,600 seconds;
   - slice: 240 seconds;
   - effective wait: 240 seconds;
   - process remains live.

3. **Budget clamp**
   - request: 3,600 seconds;
   - remaining run: 710 seconds;
   - reserve: 600 seconds;
   - effective wait: 110 seconds.

4. **Finalization reserve**
   - remaining run: 500 seconds;
   - reserve: 600 seconds;
   - wait returns immediately;
   - response contains finalization guidance.

5. **Process finishes early**
   - effective wait: 240 seconds;
   - process finishes after 3 seconds;
   - final result returns after approximately 3 seconds.

6. **No run budget**
   - budget object absent;
   - explicitly chosen compatibility behavior is verified.

### Integration test

Simulate:

```text
outer budget: 5,400 seconds
finalization reserve: 600 seconds
command: never-ending test process
configured auto-await: 240 seconds
requested process wait: 3,600 seconds
```

Verify:

- Fast-Agent periodically returns model control;
- no single wait consumes the finalization reserve;
- the process can be stopped;
- output remains inspectable;
- the model receives remaining-budget metadata; and
- the outer timeout is not the first component to signal budget exhaustion.

## Benchmark evaluation plan

Static and unit tests should precede any paid benchmark.

If a model evaluation is later authorized, use a small prospectively frozen,
matched-concurrency panel with interleaved arms:

```text
control:   foreground auto-await disabled or 10 seconds
treatment: foreground auto-await 240 seconds
budgeted:  foreground auto-await 240 seconds plus sliced budget-aware waits
```

Record:

- task reward and partial tests;
- tool and provider calls;
- auto-await and process-wait time;
- time to first model control after command start;
- outer timeouts;
- commit/finalization success;
- peak context and compactions; and
- provider retries.

The evaluation should include intentionally hanging and long-but-finite
commands. A panel containing only short successful commands validates turn
reduction but cannot validate long-tail safety.
