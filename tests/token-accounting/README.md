# Local token-accounting suite

This suite validates the three artifacts produced by a real fast-agent run:

1. persisted `fast-agent-home/sessions/<id>/` state and histories;
2. an exported Codex rollout JSONL;
3. an exported ATIF v1.7 trajectory.

The inspector cross-checks per-turn Codex token records against ATIF metrics,
validates complete ATIF aggregates, checks cache/reasoning subset invariants,
and correlates turns and tool calls.

## Deterministic tests

```bash
uv run pytest -q tests/unit/scripts/test_validate_token_accounting.py
```

The deterministic test creates a typed session, exports both formats through
`SessionTraceExporter`, and validates the resulting artifacts without provider
credentials.

## Inspect an existing run

```bash
uv run scripts/validate_token_accounting.py \
  --session /path/to/fast-agent-home/sessions/<session-id> \
  --codex /path/to/session.codex.jsonl \
  --atif /path/to/session.atif.json \
  --require-tool \
  --require-cache
```

For a reasoning-disabled GPT-5.6 run:

```bash
uv run scripts/validate_token_accounting.py \
  --session /path/to/session \
  --codex /path/to/session.codex.jsonl \
  --atif /path/to/session.atif.json \
  --expect-reasoning none
```

## Live CLI matrix

By default the live runner executes two multi-turn GPT-5.6 scenarios:

```text
codexresponses.gpt-5.6-terra?reasoning=low
codexresponses.gpt-5.6-terra?reasoning=none
```

Each scenario:

- sends a cacheable prompt above normal provider minimums;
- requests one real shell tool call;
- resumes the persisted session for a second turn;
- retains stdout, stderr, commands, session logs, and runtime ATIF files;
- exports Codex and ATIF traces from the persisted session;
- validates and cross-checks all artifacts.

Run:

```bash
uv run scripts/run_token_accounting_live.py --require-cache
```

Add the major provider routes:

```bash
uv run scripts/run_token_accounting_live.py \
  --major-models \
  --require-cache
```

Or select exact model references:

```bash
uv run scripts/run_token_accounting_live.py \
  --scenario sonnet-cache=sonnet \
  --scenario gpt56-none='codexresponses.gpt-5.6-terra?reasoning=none'
```

Artifacts default to:

```text
.artifacts/token-accounting/<timestamp>/
```

`--require-cache` is intentionally optional. Use it for controlled cache
verification. Omit it when diagnosing a route whose rendered prefix or provider
cache behavior is not yet stable.

## Harbor local-Docker matrix

Prerequisites:

- Docker;
- a Harbor checkout at `~/source/harbor`, or `--harbor-root`;
- provider credentials or local Codex authentication;
- the pinned Terminal-Bench 2.1 dataset available to Harbor.

The Harbor runner builds the current fast-agent working tree as a wheel and
passes it through Harbor's `wheel_path`. It does not install the latest
published package.

The default matrix runs `cancel-async-tasks` once for:

```text
Anthropic Sonnet
GPT-5.6 reasoning low
GPT-5.6 reasoning none
Gemini
Kimi through HF
xAI Grok 4.5 reasoning low
```

Run:

```bash
uv run scripts/run_token_accounting_harbor.py
```

Use Daytona when local Docker is unavailable:

```bash
uv run scripts/run_token_accounting_harbor.py \
  --environment daytona
```

Use the smaller general tool task if required:

```bash
uv run scripts/run_token_accounting_harbor.py \
  --task log-summary-date-ranges
```

Run one route while iterating:

```bash
uv run scripts/run_token_accounting_harbor.py \
  --scenario gpt56-none='codexresponses.gpt-5.6-terra?reasoning=none'
```

Artifacts default to:

```text
.artifacts/token-accounting-harbor/<timestamp>/
```

Each Harbor trial retains:

```text
agent/trajectory.json
agent/fast-agent-home/sessions/
agent/token-accounting/session.codex.jsonl
agent/token-accounting/session.atif.json
agent/token-accounting/report.json
```

Task reward is not an accounting assertion. Artifact validation runs whenever a
trajectory and persisted session were produced.
