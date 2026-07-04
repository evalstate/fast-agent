# Skills and Execution Environments

Status: implemented on `feat/agent-envs`

## The rule

**Skills are part of the environment's filesystem.** Run-time skill discovery
scans the *active environment's* filesystem, so every path rendered into
`<available_skills>` is readable by the environment `read_text_file` tool and
every `scripts/` entry is executable by the environment shell. There is no
host-to-environment path mapping and no mirroring lifecycle: if you want skills
in a non-local environment, mount or copy them there.

Invariant (pinned by tests): every skill path the model sees must be resolvable
by a tool the model has.

## Behavior by environment

| Environment | Skill discovery | Prompt paths | Read tool | Scripts |
| --- | --- | --- | --- | --- |
| Local (default) | Host `SkillRegistry` scan, including fast-agent home resolution | Host paths | `read_text_file` (or `read_skill` when reads are disabled) | Executable |
| Docker / HF / custom with `EnvironmentFilesystem` | `scan_environment_skills()` through the environment fs, after `open()` | Environment paths | Environment `read_text_file` | Executable |
| Shell-only environment (no filesystem) | None — startup warning if host discovery found skills | — | — | — |
| ACP | Server-side environment is local, so host scan applies (same rule, degenerate case) | Host paths | `read_skill` fallback where the ACP filesystem shadows host reads | Not executable on client terminals |

## The recipe

Default discovery paths (`.fast-agent/skills`, `.agents/skills`,
`.claude/skills`) resolve against the **environment cwd**. Mounting the
fast-agent home at the environment cwd lights up skills with zero extra
configuration:

```yaml
environments:
  ubuntu:
    type: docker
    image: ubuntu:24.04
    mounts:
      - source: .
        target: /workspace
        mode: rw
      - source: .fast-agent
        target: /workspace/.fast-agent
        mode: ro
```

For copy-based environments (e.g. Hugging Face without a bucket), stage skills
with the transfer helpers before or at session start:

```python
from fast_agent.tools.environment_transfer import copy_tree

async with fast.harness(environment="hf-gpu") as harness:
    await copy_tree(harness.local, ".fast-agent/skills",
                    harness.environment, "/workspace/.fast-agent/skills")
```

(Skills are discovered when the run enters, so copy-based staging should happen
before `fast.run(...)`/`fast.harness(...)` opens, or use a bucket mount.)

## Implementation map

- `fast_agent/skills/environment_scan.py` — `scan_environment_skills()` walks
  configured (or default) directories through the `EnvironmentFilesystem`
  protocol (`list_dir`/`exists`/`read_text`); adapters need nothing new.
  Frontmatter parsing and duplicate merging are shared with the host registry
  (`SkillRegistry.parse_manifest_text`, `merge_skill_manifests`).
- `FastAgent._apply_environment_skills()` — called from `run()` and
  `AgentHarness.__aenter__` after the environment opens and before agents are
  instantiated. Local environments return early (host discovery from
  `FastAgentRunLifecycle.enter()` already applies); filesystem environments
  re-apply env-scanned manifests; shell-only environments clear manifests with
  a startup warning.
- `McpAgent.skill_read_tool_name` — two-state: `read_text_file` when the active
  filesystem runtime exposes it, else `read_skill`. The previous
  "environment runtime forces host `read_skill`" rule is deleted; environment
  manifests carry environment paths, so the environment read tool reads them
  directly.

## Edge cases

- **`shell_execution.enable_read_text_file: false`** — skills fall back to
  `read_skill`. For a *non-local* environment this combination is degraded:
  `read_skill` validates against host paths and cannot read environment-side
  manifests. Acceptable for now; revisit if anyone hits it.
- **Per-agent explicit `skills:` entries** (paths/registries/manifests in
  Python or AgentCards) resolve host-side at lifecycle enter, unchanged. With a
  non-local environment those manifests keep host paths; explicit entries are
  the caller's responsibility.
- **Live skill management** (`/skills add`, marketplace install) writes to the
  host managed directory. A live (mounted) environment sees changes
  immediately; copy-based environments do not get mid-session installs.
- **Windows hosts with remote environments** — environment manifest paths are
  posix strings stored via `Path`; rendering on a Windows host would use
  backslashes. Known limitation, not currently handled.
- Manifest mutation surfaces: TUI/ACP `/skills` handlers reload + rebuild the
  instruction; harness `env.skills.add()/replace()` updates manifests without
  an instruction rebuild (callers refresh explicitly). Unifying this is still
  open (see review notes).

## Test coverage

- `tests/unit/fast_agent/skills/test_environment_scan.py` — scanner contract
  (defaults relative to env cwd, custom directories, duplicate override
  warning, invalid manifest reporting) using `LocalEnvironment` as a real
  `EnvironmentFilesystem`.
- `tests/unit/fast_agent/core/test_environment_skills.py` — lifecycle contract:
  environment-fs skills replace host discovery; shell-only environments
  disable skills; local environments keep host discovery.
- `tests/unit/fast_agent/tools/test_environment_filesystem_runtime.py` —
  environment-discovered skills are read through the environment
  `read_text_file` tool and `read_skill` is not advertised.
