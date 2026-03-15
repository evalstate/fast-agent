# Resume Session Refactor Proposal

## Context

We currently have `SessionManager.resume_session_agents(...)` returning a positional tuple.
Historically this shape has been:

```python
(session, loaded, missing_agents)
```

A recent change introduced optional `usage_notices` handling in one caller, which led to a shape mismatch (`expected 4, got 3`) when test stubs and manager implementation returned 3 items.

This is **not** a session-file format issue; it is an in-memory API contract issue.

---

## Recommendation

Instead of hard-switching from 3-tuple to 4-tuple everywhere, move to a typed return object.

### Proposed return type

Add a dataclass in `src/fast_agent/session/session_manager.py`:

```python
@dataclass(frozen=True)
class ResumeSessionAgentsResult:
    session: Session
    loaded: dict[str, pathlib.Path]
    missing_agents: list[str]
    usage_notices: list[str] = field(default_factory=list)
```

Then change:

```python
def resume_session_agents(...) -> ResumeSessionAgentsResult | None
```

This removes fragile positional unpacking and makes extension safe.

---

## Why this is better than a hard tuple switch

A hard tuple switch to 4 items is possible, but still fragile:
- easy to regress on positional unpacking
- easy for tests/stubs to drift
- future additions repeat the same problem

A typed result gives:
- explicit field names
- safer future evolution
- cleaner callsites
- better type-checking and IDE support

---

## Refactor plan

1. **Introduce dataclass** `ResumeSessionAgentsResult`.
2. **Update manager implementation** to always return the dataclass (or `None`).
3. **Update all callers** to use attributes, not tuple unpacking:
   - `src/fast_agent/cli/runtime/agent_setup.py`
   - `src/fast_agent/commands/handlers/sessions.py`
   - `src/fast_agent/acp/server/agent_acp_server.py`
4. **Update tests/stubs** to return dataclass instances.
5. **Remove temporary tuple compatibility branches** in callers.
6. **Document contract** in module docstring/type hints.

---

## Migration strategy

### Option A (preferred): single-step internal migration
Because this is internal API usage in the repo, do all callsites in one PR and remove tuple compatibility immediately.

### Option B: short transitional window
If needed, allow callers to accept both tuple and dataclass for one release, then remove tuple fallback.

---

## Validation checklist

Run after implementation:

```bash
uv run scripts/lint.py --fix
uv run scripts/typecheck.py
uv run pytest tests/unit
```

(Operator runs e2e separately.)

---

## Definition of done

- `resume_session_agents` returns `ResumeSessionAgentsResult | None` only.
- No tuple unpacking remains for this API.
- Tests/stubs use the dataclass.
- No compatibility branch needed in runtime caller.
- All checks pass.
