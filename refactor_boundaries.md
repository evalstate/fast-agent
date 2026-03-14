# Boundary-Hardening Sprint Plan

Date: 2026-03-14

## Goal

This sprint is the focused follow-up to the completed architectural refactors.

The major orchestration and lifecycle surfaces have already been cleaned up.
This sprint is about consolidating those gains by reducing compatibility drift
and boundary mismatches.

The goal is not more large rewrites. The goal is to make the newly refactored
areas safer and less likely to regress.

## Sprint-sized set

### 1. Centralize `llm` / `basic` runtime normalization

#### Why

We already hit a real regression here during the recent refactors.

The system currently has an internal/runtime distinction where:

- `AgentType.LLM` exists for `LlmDecorator`
- `AgentType.BASIC` is the card/runtime “agent” shape

Several behaviors intentionally treat these as equivalent or near-equivalent,
but that equivalence is still too implicit.

#### Scope

Likely files:

- `src/fast_agent/core/direct_factory.py`
- `src/fast_agent/core/validation.py`
- any nearby logic still assuming only `basic`

#### Deliverable

Introduce one explicit helper for shared-runtime handling, for example:

- `_is_basic_like_agent_type(...)`
- or `_normalize_runtime_agent_type(...)`

Use it in places where shared behavior is intended.

#### Exit criteria

- no ad hoc `{"llm", "basic"}` checks scattered around
- runtime creation and validation use one shared normalization rule
- behavior is explicit rather than accidental

---

### 2. Add compatibility-boundary tests

#### Why

The recent regressions were not from the core new logic itself, but from
compatibility assumptions around it.

These are cheap, high-value tests that protect refactor work.

#### Scope

Likely test targets:

- `tests/unit/fast_agent/core/`
- possibly a small integration test if runtime behavior is easier to prove there

#### Strong candidate tests

- `AgentType.LLM` is accepted anywhere basic-like runtime behavior is expected
- custom agent compatibility:
  - `agent_class`
  - `cls`
- one or two loader/validator consistency cases if appropriate

#### Exit criteria

- known compatibility cases are intentional and tested
- future dispatch/validation refactors are less likely to break them silently

---

### 3. Align shared AgentCard rules between loader and validator

#### Why

We recently refactored:

- loader
- serializer
- validator scan path

That makes this the right moment to reduce drift between them.

#### Scope

Likely files:

- `src/fast_agent/core/agent_card_loader.py`
- `src/fast_agent/core/agent_card_validation.py`

#### Focus

Compare and align:

- allowed types
- required fields
- dependency field rules
- function tool validation assumptions
- shell / cwd rules
- message file rules

#### Preferred approach

If a rule is truly shared:

- extract a small shared constant or helper

If a difference is intentional:

- leave it separate
- document it with a test

#### Exit criteria

- obvious shared rules are defined once or clearly anchored
- remaining differences are intentional and tested
- load/validate behavior is less likely to drift over time

---

## What comes after this sprint

Once the above three items are done, the next best target is:

### `src/fast_agent/mcp/oauth_client.py::build_oauth_provider`

This is the most natural next external-boundary hardening task.

---

## Out of scope for this sprint

These are worthwhile, but not part of the focused sprint-sized set:

- deeper model/config resolution cleanup
- broader reload/scan/load behavior hardening
- larger path/placeholder unification
- more large architectural refactors

---

## Definition of success

This sprint is successful when:

- runtime alias compatibility is centralized
- compatibility cases are pinned with tests
- AgentCard loader/validator rule drift is reduced
- the recent architecture refactors are better protected against regressions
