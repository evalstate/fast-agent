---
title: Execution Environments
description: Run shell and filesystem tools in local, Docker, Hugging Face, or custom environments.
social:
  title: Execution Environments
  tagline: Set up local, containerised and remote sandboxes where harness shell commands run.
  description: Run shell and filesystem tools in local, Docker, Hugging Face, or custom environments.
  alt: fast-agent social card — Execution Environments
---

# Execution Environments

Execution environments define where your agent runs shell commands.

Configure reusable environments in your fast-agent config file, then select them by name from the Python API or
the CLI.

!!! note
    Do not confuse an execution environment with your workspace or fast-agent home.
    The workspace is the project file tree for a run; the home is fast-agent's
    local config and state root, usually `<workspace>/.fast-agent`.

By default, fast-agent uses the implicit `local` environment. To start fast-agent with local shell access simply use `fast-agent -x`. 

You can switch to Docker, a Hugging Face Sandbox, or a custom adapter without changing agent
definitions.

```python
result = await harness.shell("pwd")
print(result.stdout, result.stderr, result.exit_code)
```

## Named environments

Add `environments:` to `<home>/fast-agent.yaml`. Relative mount sources resolve
against the workspace.

```yaml
default_environment: ubuntu

environments:
  ubuntu:
    type: docker
    image: ubuntu:24.04
    shell: bash
    cwd: /workspace
    mounts:
      - source: .
        target: /workspace
        mode: rw

  hf-gpu:
    type: huggingface
    image: python:3.12
    flavor: cpu-basic
    cwd: /workspace
    volume_mounts:
      - hf://buckets/username/my-bucket:/workspace:rw

  staging:
    type: custom
    class: mycompany.envs:KubernetesEnvironment
    params:
      namespace: agents-staging
```

Use the configured name wherever an execution environment is accepted:

```python
async with fast.harness(environment="ubuntu") as harness:
    result = await harness.shell("pwd")

async with fast.run(environment="hf-gpu") as agent_app:
    await agent_app.interactive()
```

From the CLI:

```bash
fast-agent go --environment ubuntu -x
```

Omitting `environment=` uses `default_environment`; if no default is configured,
`local` is used.

## Which file tools do agents get?

File tools are workspace tools: `read_text_file`, `write_text_file`,
`edit_file`, and `apply_patch` are exposed only when they operate on the same
tree the shell sees.

| Runtime | File tools |
| ------- | ---------- |
| ACP client with file capabilities | Client workspace tools, with local gap-fill for missing edit/patch tools. |
| Active environment implements `EnvironmentFilesystem` | Environment filesystem, including local, mounted Docker, and Hugging Face Sandbox environments. |
| Plain local shell with no injected environment object | Host workspace filesystem. |
| Shell-only environment | No model-facing workspace file tools. |

There is no host fallback for a remote/container shell. If Docker runs in
`/workspace`, file tools must target the same mounted `/workspace` tree, not an
unrelated host path.

## Environment variables

fast-agent does not copy your host process environment wholesale into Docker
containers or Hugging Face Sandboxes. Environment variables are opt-in:

- `env:` on the configured environment supplies default variables for shell
  commands in that environment.
- Per-call `env` passed to `harness.shell(..., env={...})` or
  `ShellExecutionRequest.env` is merged over the configured defaults for that
  command.
- Provider credentials used by fast-agent itself, such as the Hugging Face
  `token:` used to create a Sandbox, are not automatically exposed inside the
  environment as command environment variables.

Use `${...}` references with `fast-agent.secrets.yaml` or host environment
variables when you intentionally want a value to be sent into the execution
environment:

```yaml
environments:
  ubuntu:
    type: docker
    image: ubuntu:24.04
    cwd: /workspace
    env:
      APP_ENV: development
      API_TOKEN: ${MY_APP_TOKEN}

  hf:
    type: huggingface
    image: python:3.12
    cwd: /workspace
    token: ${HF_TOKEN}        # used by fast-agent to create/manage the Sandbox
    env:
      APP_ENV: development
      DATASET_TOKEN: ${MY_DATASET_TOKEN}  # visible to commands in the Sandbox
```

```python
async with fast.harness(environment="ubuntu") as harness:
    result = await harness.shell(
        "printf '%s\n' \"$APP_ENV:$RUN_ID\"",
        env={"RUN_ID": "manual-test"},
    )
```

Treat anything under `env:` as visible to commands running in that environment.

## Skills follow the environment

Agent Skills are discovered from the active environment's filesystem, so skill
paths shown to the model are always readable by its file tools and skill
scripts are executable by its shell. Local runs scan the host as usual. For a
non-local environment, mount (or copy) your skills into it — the default
discovery paths (`.fast-agent/skills`, `.agents/skills`, `.claude/skills`)
resolve against the environment working directory:

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

An environment without a filesystem cannot surface skills; if skills are
configured, fast-agent warns at startup and runs without them.

In the interactive UI, `/skills` lists locally installed skills. When the active
environment is remote or containerized, fast-agent adds a warning because those
local skills may not be present in the environment. Use `/system` to inspect the
resolved prompt and confirm which skills the agent can actually read.

## Copying files between environments

Harness code always has a host-side local environment at `harness.local`
alongside the active environment. Use transfer helpers for explicit staging and
artifact collection:

```python
from fast_agent.tools.environment_transfer import copy_tree

async with fast.harness(environment="hf-gpu") as harness:
    await copy_tree(harness.local, "datasets/input", harness.environment, "/workspace/input")

    session = await harness.session("train", agent_name="researcher")
    await session.generate("Train on /workspace/input and write metrics to /workspace/out")

    await copy_tree(harness.environment, "/workspace/out", harness.local, "results")
```

!!! note

    "Sandbox" is adapter-specific. Docker and remote providers may offer useful
    isolation, but fast-agent does not claim a universal sandbox security
    contract. Treat mounted files, credentials, and network access as explicit
    product choices.

## Defining environments in code

Pass a `ShellEnvironment` to `FastAgent.harness(...)` for programmatic shell
calls. If the object also implements `EnvironmentFilesystem`, shell-enabled agents
created under the harness use the same environment for model-facing file tools.

```python
from pathlib import Path

from fast_agent.tools.docker_shell_environment import (
    DockerManagedShellEnvironment,
    DockerMount,
)

environment = DockerManagedShellEnvironment(
    image="ubuntu:24.04",
    shell="bash",
    cwd="/workspace",
    mounts=[DockerMount(source=Path.cwd(), target="/workspace", mode="rw")],
)

async with fast.harness(environment=environment) as harness:
    result = await harness.shell("pwd")
    assert result.stdout.strip() == "/workspace"
```

The harness calls `environment.open()` on entry and `environment.close()` on
exit. `HarnessSession.shell(...)` delegates to the same environment.

## Interactive agents

You can also pass an environment to `FastAgent.run(...)`. This affects
model-facing shell tools for shell-enabled agents, including normal interactive
sessions.

```python
async with fast.run(environment=environment) as agent_app:
    await agent_app.interactive()
```

If an agent exposes the `execute` tool, model tool calls run in the injected
environment. This lets you use the normal fast-agent interactive UI while keeping
shell commands inside Docker or another adapter.

Manual example:

```bash
uv run python examples/docker-shell/interactive.py
```

## Docker environments

fast-agent includes Docker shell adapters and a mounted environment adapter:

| Adapter | Use when |
| ------- | -------- |
| `DockerManagedShellEnvironment` | fast-agent should start and remove a disposable container for the harness run. |
| `DockerShellEnvironment` | You already have a running container and want fast-agent to execute commands inside it. |
| `DockerMountedEnvironment` | You want Docker `execute` and model-facing file tools to target the same bind-mounted workspace. |

Managed Docker example:

```python
from pathlib import Path

from fast_agent.tools.docker_shell_environment import (
    DockerManagedShellEnvironment,
    DockerMount,
)

environment = DockerManagedShellEnvironment(
    image="ubuntu:24.04",
    cwd="/workspace",
    mounts=[DockerMount(Path.cwd(), "/workspace")],
)
```

Mounted environment example:

```python
from pathlib import Path

from fast_agent.tools.docker_shell_environment import DockerMountedEnvironment

environment = DockerMountedEnvironment(
    image="ubuntu:24.04",
    workspace=Path.cwd(),
    target="/workspace",
)
```

Existing container example:

```python
from fast_agent.tools.docker_shell_environment import DockerShellEnvironment

environment = DockerShellEnvironment(
    container="fast-agent-workspace",
    cwd="/workspace",
    shell="bash",
)
```

The Docker adapters run commands with `docker exec -w <cwd> ... <shell> -lc
<command>` for POSIX-style shells, or PowerShell flags for `pwsh`/`powershell`.
Environment variables passed to `harness.shell(..., env={...})` are forwarded
with Docker `-e NAME` arguments and inherited from the Docker CLI subprocess
environment, so values are not placed directly in the command argv. Managed
Docker containers are started without the configured `env:` on their long-lived
`sleep infinity` process; the configured and per-command variables are applied
to each `docker exec` command. `DockerMountedEnvironment` maps file tool paths
under `target` back to the host bind mount, so `execute`, `read_text_file`, and
`apply_patch` all operate on the same visible tree.

## Hugging Face Sandbox environments

`HuggingFaceSandboxEnvironment` runs both shell execution and model-facing file
tools inside a Hugging Face Sandbox. Configured environments can mount Hub
volumes with the same `hf://...:/mount/path[:ro|:rw]` syntax used by the Hub CLI:

```yaml
environments:
  hf-gpu:
    type: huggingface
    image: python:3.12
    flavor: cpu-basic
    cwd: /workspace
    volume_mounts:
      - hf://buckets/username/my-bucket:/workspace:rw
      - hf://datasets/username/reference-data:/data:ro
```

At the Python API layer, bucket mounts can also be constructed explicitly:

```python
from fast_agent.tools.huggingface_sandbox_environment import (
    HuggingFaceBucketMount,
    HuggingFaceSandboxEnvironment,
)

environment = HuggingFaceSandboxEnvironment(
    image="python:3.12",
    flavor="cpu-basic",
    cwd="/workspace",
    bucket_mounts=(
        HuggingFaceBucketMount(
            source="username/my-bucket",
            mount_path="/workspace",
            read_only=False,
        ),
        HuggingFaceBucketMount(
            source="username/reference-data",
            mount_path="/data",
            read_only=True,
        ),
    ),
)
```

Manual example:

```bash
uv run python examples/huggingface-sandbox/interactive.py \
  --bucket username/my-bucket:/workspace:rw
```

The config surface currently creates dedicated Sandboxes with
`huggingface_hub.Sandbox.create`. Hugging Face `SandboxPool` pooling is not
exposed through `fast-agent.yaml`; if you need pooled sandbox lifecycle today,
wrap the Hub pool in a custom `ShellEnvironment`/`EnvironmentFilesystem`
adapter and configure it as `type: custom`.

For Hugging Face Sandboxes, configured `env:` is passed to
`Sandbox.create(env=...)` and also merged into each `sandbox.run(...)` call.
Per-command env overrides are added for that command only. The `token:` field is
used by fast-agent to authenticate with the Hub API; it is not automatically
forwarded as `HF_TOKEN` inside the Sandbox. If commands inside the Sandbox need
`HF_TOKEN`, add it explicitly under `env:`.

## Docker and Hugging Face examples

There are example configurations and programs in `examples/environments`. 

Docker is not required for the automated test suite. To manually exercise the
managed Docker adapter:

```bash
uv run python examples/docker-shell/docker_shell_harness.py
```

The example creates a temporary host workspace, mounts it into an Ubuntu
container at `/workspace`, runs `harness.shell(...)`, and verifies a container
write is visible on the host.

## CWD semantics

`cwd` is a string at the environment protocol boundary:

- local environments interpret it as a host path;
- Docker environments interpret it as a container path;
- remote environments should interpret it as a provider-side path.

This keeps container paths like `/workspace` from being coerced into host
`Path` objects.

Environment objects may be shared across agents and harness sessions. Treat
`cwd` as adapter-level default state and pass per-agent or per-call working
directories through `AgentConfig.cwd`, `harness.shell(..., cwd=...)`, or
`ShellExecutionRequest.cwd`.

## Implementing a custom environment

Custom environments implement `ShellEnvironment` from
`fast_agent.tools.execution_environment`.

```python
from fast_agent.tools.execution_environment import (
    EnvironmentFileEntry,
    ShellEnvironment,
    ShellExecution,
    ShellExecutionOptions,
    ShellExecutionRequest,
    ShellExecutionResult,
    ShellRuntimeInfo,
)


class MyEnvironment:
    async def open(self) -> None:
        ...

    @property
    def cwd(self) -> str:
        return "/workspace"

    def runtime_info(self) -> ShellRuntimeInfo:
        return ShellRuntimeInfo(name="bash", kind="remote", provider="my-provider")

    async def execute(self, request: ShellExecutionRequest, *, callbacks=None) -> ShellExecution:
        result = await my_provider_exec(
            request.command,
            cwd=request.cwd or self.cwd,
            env=dict(request.env or {}),
            timeout=request.timeout,
        )
        return ShellExecution(
            result=ShellExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
            ),
            options=ShellExecutionOptions(timeout_seconds=request.timeout),
        )

    async def close(self) -> None:
        ...
```

Adapters that can stream output should call callback hooks as output arrives:

- `on_stdout(text)`
- `on_stderr(text)`
- `on_idle_warning(elapsed, remaining)`
- `on_timeout()`

If a provider cannot stream, ignore callbacks and return final output.

To give the LLM natural file access to the same environment, implement
`EnvironmentFilesystem` on the same object:

```python
from fast_agent.tools.execution_environment import EnvironmentFilesystem


class MyEnvironment:
    # ShellEnvironment methods above...

    def resolve_path(self, path: str) -> str:
        ...

    async def read_text(self, path: str) -> str:
        ...

    async def write_text(self, path: str, content: str) -> None:
        ...

    async def exists(self, path: str) -> bool:
        ...

    async def list_dir(self, path: str) -> list[EnvironmentFileEntry]:
        ...

    async def mkdir(self, path: str) -> None:
        ...

    async def remove(self, path: str) -> None:
        ...
```

Keep provider-specific concepts inside the adapter. For example, Hugging Face
bucket mounts belong on `HuggingFaceSandboxEnvironment`, while the generic
runtime only depends on `ShellEnvironment` and `EnvironmentFilesystem`.

`ShellRuntimeInfo.kind` is coarse display metadata. Built-in values include
`local`, `docker`, and `remote`, but custom providers can use another stable
string and should set `provider` to the adapter name.

For Skills, fast-agent scans the active environment filesystem when one is
available, and formats skill paths for the environment `read_text_file` tool.
Local runs scan the host workspace/home paths. If an environment has shell
execution but no filesystem contract, Skills are not surfaced for that run.
