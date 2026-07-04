<!--
  GENERATED FILE — DO NOT EDIT.
  Source: generate_reference_docs.py / fast_agent.tools.environment_config
-->

# Execution Environment Configuration

Use this resource when creating or editing `fast-agent.yaml` named execution environments. The field reference below is generated from the Pydantic config models so schema changes are reflected here.

## Top-Level Shape

```yaml
default_environment: local

environments:
  local:
    type: local
    cwd: .

  ubuntu:
    type: docker
    image: ubuntu:24.04
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
      - hf://datasets/username/reference-data:/data:ro
```

## Rules

- `local` is always available implicitly.
- `default_environment` must name `local` or a configured environment.
- Environment names starting with `_` are reserved.
- Environment specs reject unknown fields.
- Docker specs require exactly one of `image` or `container`.
- Docker mount sources are resolved against the workspace root; use `mounts`, not volume flags in `docker_args`.
- Hugging Face `volume_mounts` use `hf://[models|datasets|spaces|buckets]/namespace/name[/path]:/mount/path[:ro|:rw]`; omitted type defaults to models.
- Hugging Face Sandbox pooling (`SandboxPool`) is not exposed in `fast-agent.yaml`; use a custom environment adapter for pooled sandbox lifecycle.
- Put tokens/secrets in `fast-agent.secrets.yaml` or environment variables.

## Field Reference

### Local Environment

| Field | Type | Default | Description | Examples |
| --- | --- | --- | --- | --- |
| `type` | `Literal['local']` | `'local'` | Use the host local shell. |  |
| `cwd` | `str \| None` | `None` | Working directory for local shell and file tools. Relative paths resolve against the workspace root. | `'.'` |
| `env` | `dict[str, str]` | `{}` | Environment variables applied to shell execution. | `{'PYTHONUNBUFFERED': '1'}` |

### Docker Environment

| Field | Type | Default | Description | Examples |
| --- | --- | --- | --- | --- |
| `type` | `Literal['docker']` | `required` | Run shell commands in Docker or a Docker-compatible CLI. |  |
| `image` | `str \| None` | `None` | Container image to start. Provide exactly one of `image` or `container`. | `'ubuntu:24.04'` |
| `container` | `str \| None` | `None` | Existing container name or ID to execute in. Provide exactly one of `image` or `container`. | `'fast-agent-ci'` |
| `container_cli` | `str` | `'docker'` | Executable used for container operations, for example `docker` or `wslc`. | `'docker'`, `'wslc'` |
| `shell` | `str` | `'bash'` | Shell executable used inside the container. | `'bash'`, `'sh'`, `'pwsh'` |
| `cwd` | `str` | `'/workspace'` | Working directory inside the container. | `'/workspace'` |
| `mounts` | `list[fast_agent.tools.environment_config.EnvironmentMountSpec]` | `[]` | Docker bind mounts. Use this instead of volume flags in `docker_args`. |  |
| `env` | `dict[str, str]` | `{}` | Environment variables applied to shell execution. |  |
| `docker_args` | `list[str]` | `[]` | Extra container creation arguments. Volume and lifecycle flags are rejected; use `mounts` for bind mounts. | `['--network=none']` |

### Docker Mount

| Field | Type | Default | Description | Examples |
| --- | --- | --- | --- | --- |
| `source` | `str` | `required` | Host workspace path to bind mount. Relative paths resolve against the workspace root. | `'.'` |
| `target` | `str` | `required` | Absolute path inside the Docker container. | `'/workspace'` |
| `mode` | `Literal['ro', 'rw']` | `'rw'` | Docker bind mount access mode: read-only (`ro`) or read-write (`rw`). | `'rw'` |

### Hugging Face Environment

| Field | Type | Default | Description | Examples |
| --- | --- | --- | --- | --- |
| `type` | `Literal['huggingface']` | `required` | Run shell commands in a Hugging Face Sandbox. |  |
| `image` | `str` | `'python:3.12'` | Sandbox container image. | `'python:3.12'` |
| `flavor` | `str` | `'cpu-basic'` | Hugging Face Sandbox hardware flavor. | `'cpu-basic'` |
| `cwd` | `str` | `'/workspace'` | Working directory inside the sandbox. | `'/workspace'` |
| `bucket_mounts` | `list[fast_agent.tools.environment_config.HuggingFaceBucketMountSpec]` | `[]` | Legacy bucket-only mount shorthand. Prefer `volume_mounts` for new config. |  |
| `volume_mounts` | `list[fast_agent.tools.environment_config.HuggingFaceVolumeMountSpec]` | `[]` | Hugging Face Sandbox volume mounts using `hf://...:/mount/path[:ro\|:rw]` syntax. |  |
| `env` | `dict[str, str]` | `{}` | Environment variables applied to shell execution. |  |
| `token` | `str \| None` | `None` | Hugging Face token. Prefer `fast-agent.secrets.yaml` or `${HF_TOKEN}`. | `'${HF_TOKEN}'` |

### Hugging Face Volume Mount

| Field | Type | Default | Description | Examples |
| --- | --- | --- | --- | --- |
| `uri` | `str` | `required` | Hugging Face mount URI: `hf://[models\|datasets\|spaces\|buckets]/namespace/name[/path]:/mount/path[:ro\|:rw]`. The type defaults to models. | `'hf://buckets/username/my-bucket:/workspace:rw'`, `'hf://datasets/username/reference-data:/data:ro'` |

### Hugging Face Bucket Mount Shorthand

| Field | Type | Default | Description | Examples |
| --- | --- | --- | --- | --- |
| `source` | `str` | `required` | Hugging Face bucket identifier in `namespace/name` form. | `'username/my-bucket'` |
| `mount_path` | `str` | `required` | Absolute path where the bucket is mounted inside the sandbox. | `'/workspace'` |
| `read_only` | `bool` | `False` | Whether the bucket mount is read-only. | `false` |
| `path` | `str \| None` | `None` | Optional subfolder prefix inside the bucket to mount. | `'subdir'` |

### Custom Environment

| Field | Type | Default | Description | Examples |
| --- | --- | --- | --- | --- |
| `type` | `Literal['custom']` | `required` | Load a custom ShellEnvironment class. |  |
| `class` | `str` | `required` | Import path in `module.path:ClassName` format. | `'mycompany.envs:KubernetesEnvironment'` |
| `params` | `dict[str, Any]` | `{}` | Keyword arguments passed to the custom environment class. |  |

