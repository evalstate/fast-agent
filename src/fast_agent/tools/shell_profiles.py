from typing import Literal

type ResolvedShellToolProfile = Literal[
    "native",
    "minimal_process",
    "grok_shell",
    "luna_exec",
]
type ShellToolProfile = Literal["auto"] | ResolvedShellToolProfile


def resolve_shell_tool_profile(
    configured_profile: ShellToolProfile,
    model_profile: ResolvedShellToolProfile | None,
) -> ResolvedShellToolProfile:
    if configured_profile == "auto":
        return model_profile or "minimal_process"
    return configured_profile
