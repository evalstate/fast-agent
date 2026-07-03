---
title: Agent Environments
description: Run shell and filesystem tools in local, Docker, Hugging Face, or custom environments.
social:
  title: Agent Environments
  tagline: Swap where harness shell commands run.
  description: Run shell and filesystem tools in local, Docker, Hugging Face, or custom environments.
  alt: fast-agent social card — Agent Environments
---

# Agent Environments

Agent environments define where shell commands run. If the environment also
implements `EnvironmentFilesystem`, model-facing file tools such as
`read_text_file`, `write_text_file`, `edit_file`, and `apply_patch` use that
same environment filesystem.

By default, fast-agent uses the local shell and local filesystem helper tools.
You can inject another environment, such as Docker or a Hugging Face Sandbox,
without changing agent definitions.

```python
result = await harness.shell("pwd")
print(result.stdout, result.stderr, result.exit_code)
```

!!! note

    "Sandbox" is adapter-specific. Docker and remote providers may offer useful
    isolation, but fast-agent does not claim a universal sandbox security
    contract. Treat mounted files, credentials, and network access as explicit
    product choices.

## Harness environment injection

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
with Docker `-e` arguments. `DockerMountedEnvironment` maps file tool
paths under `target` back to the host bind mount, so `execute`, `read_text_file`,
and `apply_patch` all operate on the same visible tree.

## Hugging Face Sandbox environments

`HuggingFaceSandboxEnvironment` runs both shell execution and model-facing file
tools inside a Hugging Face Sandbox. Bucket mounts are provider-specific product
surface on this adapter, so generic fast-agent code does not need to know about
Hugging Face volumes.

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

## Manual Docker smoke test

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
`fast_agent.tools.session_environment`.

```python
from fast_agent.tools.session_environment import (
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
from fast_agent.tools.session_environment import EnvironmentFilesystem


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

For Skills, fast-agent currently discovers installed Skill manifests from the
host fast-agent home. When an injected `EnvironmentFilesystem` is active,
those host-path Skill files are still read through `read_skill`; environment
file tools remain available for files inside the Docker container, sandbox, or
remote workspace.
