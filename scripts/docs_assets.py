#!/usr/bin/env python3
"""Build and record committed documentation assets.

Usage:
    uv run scripts/docs_assets.py list
    uv run scripts/docs_assets.py check
    uv run scripts/docs_assets.py build
    uv run scripts/docs_assets.py record tui-shell
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs" / "docs"
ASSETS = ROOT / "docs" / "docs" / "assets"
VENDOR_ASCIINEMA = ASSETS / "vendor" / "asciinema-player"
ASCIINEMA_INDEX = ASSETS / "asciinema-index.json"


@dataclass(frozen=True)
class TerminalCastScenario:
    name: str
    title: str
    output: Path
    cols: int
    rows: int
    idle_time_limit: float
    prompt: str
    shell_command: str
    notes: str = ""


@dataclass(frozen=True)
class CastRecorder:
    name: str
    command: str
    notes: str = ""


def _tui_shell_scenario() -> TerminalCastScenario:
    model = os.environ.get("FAST_AGENT_TUI_DEMO_MODEL", "deepseek")
    command = os.environ.get("FAST_AGENT_TUI_DEMO_COMMAND")
    if command is None:
        command = f"fast-agent -x --model {model}"
    return TerminalCastScenario(
        name="tui-shell",
        title="fast-agent TUI shell commands",
        output=ASSETS / "tui" / "tui-shell.cast",
        cols=int(os.environ.get("FAST_AGENT_TUI_DEMO_COLS", "96")),
        rows=int(os.environ.get("FAST_AGENT_TUI_DEMO_ROWS", "22")),
        idle_time_limit=float(os.environ.get("FAST_AGENT_TUI_DEMO_IDLE_TIME_LIMIT", "1.3")),
        prompt=os.environ.get("FAST_AGENT_TUI_DEMO_PROMPT", "Good morning"),
        shell_command=command,
    )


def _model_picker_scenario() -> TerminalCastScenario:
    command = os.environ.get("FAST_AGENT_MODEL_PICKER_DEMO_COMMAND", "fast-agent go")
    return TerminalCastScenario(
        name="model-picker",
        title="fast-agent model picker",
        output=ASSETS / "models" / "model-picker.cast",
        cols=int(os.environ.get("FAST_AGENT_MODEL_PICKER_DEMO_COLS", "96")),
        rows=int(os.environ.get("FAST_AGENT_MODEL_PICKER_DEMO_ROWS", "21")),
        idle_time_limit=float(
            os.environ.get("FAST_AGENT_MODEL_PICKER_DEMO_IDLE_TIME_LIMIT", "1.3")
        ),
        prompt="",
        shell_command=command,
    )


def _skills_direct_install_scenario() -> TerminalCastScenario:
    return TerminalCastScenario(
        name="skills-direct-install",
        title="fast-agent direct skill install",
        output=ASSETS / "tui" / "skills-direct-install.cast",
        cols=int(os.environ.get("FAST_AGENT_SKILLS_DEMO_COLS", "96")),
        rows=int(os.environ.get("FAST_AGENT_SKILLS_DEMO_ROWS", "20")),
        idle_time_limit=float(os.environ.get("FAST_AGENT_SKILLS_DEMO_IDLE_TIME_LIMIT", "1.3")),
        prompt="",
        shell_command="",
    )


def _skills_slash_commands_scenario() -> TerminalCastScenario:
    command = os.environ.get(
        "FAST_AGENT_SKILLS_SLASH_DEMO_COMMAND",
        "fast-agent -x --model passthrough",
    )
    return TerminalCastScenario(
        name="skills-slash-commands",
        title="fast-agent skills slash commands",
        output=ASSETS / "tui" / "skills-slash-commands.cast",
        cols=int(os.environ.get("FAST_AGENT_SKILLS_SLASH_DEMO_COLS", "96")),
        rows=int(os.environ.get("FAST_AGENT_SKILLS_SLASH_DEMO_ROWS", "24")),
        idle_time_limit=float(
            os.environ.get("FAST_AGENT_SKILLS_SLASH_DEMO_IDLE_TIME_LIMIT", "1.3")
        ),
        prompt="",
        shell_command=command,
    )


def _skills_over_mcp_scenario() -> TerminalCastScenario:
    command = os.environ.get(
        "FAST_AGENT_SKILLS_MCP_DEMO_COMMAND",
        "fast-agent -x --model passthrough",
    )
    return TerminalCastScenario(
        name="skills-over-mcp",
        title="fast-agent Skills over MCP",
        output=ASSETS / "tui" / "skills-over-mcp.cast",
        cols=int(os.environ.get("FAST_AGENT_SKILLS_MCP_DEMO_COLS", "96")),
        rows=int(os.environ.get("FAST_AGENT_SKILLS_MCP_DEMO_ROWS", "22")),
        idle_time_limit=float(os.environ.get("FAST_AGENT_SKILLS_MCP_DEMO_IDLE_TIME_LIMIT", "1.3")),
        prompt="",
        shell_command=command,
        notes=(
            "Live Hugging Face stable-v2 connection; demonstrates the SEP-2640 "
            "Skills Extension Draft (d7490ecd) resource-manifest workflow, not "
            "legacy index/archive servers."
        ),
    )


def _mcp_tool_schema_scenario() -> TerminalCastScenario:
    command = os.environ.get(
        "FAST_AGENT_TOOL_SCHEMA_DEMO_COMMAND",
        "fast-agent --model passthrough",
    )
    return TerminalCastScenario(
        name="mcp-tool-schema",
        title="fast-agent MCP structured output schema",
        output=ASSETS / "tui" / "mcp-tool-schema.cast",
        cols=int(os.environ.get("FAST_AGENT_TOOL_SCHEMA_DEMO_COLS", "110")),
        rows=int(os.environ.get("FAST_AGENT_TOOL_SCHEMA_DEMO_ROWS", "32")),
        idle_time_limit=float(os.environ.get("FAST_AGENT_TOOL_SCHEMA_DEMO_IDLE_TIME_LIMIT", "3.0")),
        prompt="",
        shell_command=command,
        notes=(
            "Live MCP connection; defaults to the Hugging Face MCP server at "
            "http://localhost:3000/mcp and inspects hf_whoami."
        ),
    )


def _hf_image_generation_scenario() -> TerminalCastScenario:
    command = os.environ.get(
        "FAST_AGENT_HF_IMAGE_DEMO_COMMAND",
        "fast-agent -x --model codexplan",
    )
    prompt = os.environ.get(
        "FAST_AGENT_HF_IMAGE_DEMO_PROMPT",
        (
            "Use Z-Image to generate a wide cinematic landscape: "
            "a quiet alpine lake at sunrise, "
            "dark pine silhouettes, snow-capped mountains, warm orange sky reflected "
            "in the water, bold simple shapes, high contrast, no text"
        ),
    )
    return TerminalCastScenario(
        name="hf-image-generation",
        title="fast-agent Hugging Face image generation",
        output=ASSETS / "tui" / "hf-image-generation.cast",
        cols=int(os.environ.get("FAST_AGENT_HF_IMAGE_DEMO_COLS", "120")),
        rows=int(os.environ.get("FAST_AGENT_HF_IMAGE_DEMO_ROWS", "34")),
        idle_time_limit=float(os.environ.get("FAST_AGENT_HF_IMAGE_DEMO_IDLE_TIME_LIMIT", "1.3")),
        prompt=prompt,
        shell_command=command,
    )


def _elicitation_sandbox_scenario() -> TerminalCastScenario:
    return TerminalCastScenario(
        name="elicitation-sandbox",
        title="Modern request-scoped MCP elicitation",
        output=ASSETS / "mcp" / "elicitation-sandbox.cast",
        cols=int(os.environ.get("FAST_AGENT_ELICITATION_DEMO_COLS", "96")),
        rows=int(os.environ.get("FAST_AGENT_ELICITATION_DEMO_ROWS", "24")),
        idle_time_limit=float(os.environ.get("FAST_AGENT_ELICITATION_DEMO_IDLE_TIME_LIMIT", "1.3")),
        prompt="",
        shell_command="uv run sandbox_demo.py",
        notes="Runs the simulated t4-small quickstart with modern MCP request retries.",
    )


def _mcp_inspect_legacy_scenario() -> TerminalCastScenario:
    return TerminalCastScenario(
        name="mcp-inspect-legacy",
        title="fast-agent legacy remote MCP status",
        output=ASSETS / "mcp" / "mcp-inspect-legacy.cast",
        cols=int(os.environ.get("FAST_AGENT_MCP_LEGACY_DEMO_COLS", "112")),
        rows=int(os.environ.get("FAST_AGENT_MCP_LEGACY_DEMO_ROWS", "30")),
        idle_time_limit=float(os.environ.get("FAST_AGENT_MCP_LEGACY_DEMO_IDLE_TIME_LIMIT", "1.3")),
        prompt="",
        shell_command="fast-agent -x --model passthrough",
        notes=(
            "Deterministic local legacy Streamable HTTP server; demonstrates "
            "MCP-Session-Id capture, health pings, and a high-resolution timeline."
        ),
    )


def _scenarios() -> dict[str, TerminalCastScenario]:
    scenarios = [
        _tui_shell_scenario(),
        _model_picker_scenario(),
        _skills_direct_install_scenario(),
        _skills_slash_commands_scenario(),
        _skills_over_mcp_scenario(),
        _mcp_tool_schema_scenario(),
        _hf_image_generation_scenario(),
        _elicitation_sandbox_scenario(),
        _mcp_inspect_legacy_scenario(),
    ]
    return {scenario.name: scenario for scenario in scenarios}


def _a2a_recorders() -> dict[Path, CastRecorder]:
    a2a_assets = ASSETS / "a2a"
    return {
        (a2a_assets / "a2a-client-input-required.cast").relative_to(ROOT): CastRecorder(
            name="a2a-client-input-required",
            command="uv run scripts/a2a_docs_pipeline.py record",
            notes="Deterministic turn-continuation transcript; regenerated with the A2A batch.",
        ),
        (a2a_assets / "a2a-server-card.cast").relative_to(ROOT): CastRecorder(
            name="a2a-server-card",
            command="uv run scripts/a2a_docs_pipeline.py record",
            notes="Deterministic A2A server card recording; regenerated with the A2A batch.",
        ),
    }


def _cast_recorders() -> dict[Path, CastRecorder]:
    recorders = {
        scenario.output.relative_to(ROOT): CastRecorder(
            name=scenario.name,
            command=f"uv run scripts/docs.py cast-build {scenario.name}",
            notes=scenario.notes,
        )
        for scenario in _scenarios().values()
    }
    recorders.update(_a2a_recorders())
    return recorders


_ATTR_RE = re.compile(r"""(?P<name>data-fa-[\w-]+|class)=["'](?P<value>[^"']*)["']""")
_DEMO_RE = re.compile(r"<div\b(?P<attrs>[^>]*\bdata-fa-asciinema-cast=[^>]*)>", re.DOTALL)


def _attrs(fragment: str) -> dict[str, str]:
    return {match.group("name"): match.group("value") for match in _ATTR_RE.finditer(fragment)}


def _resolve_docs_asset_reference(page: Path, value: str) -> Path:
    if "assets/" in value:
        return (DOCS / value[value.index("assets/") :]).relative_to(ROOT)
    resolved = (page.parent / value).resolve()
    try:
        return resolved.relative_to(ROOT)
    except ValueError:
        return resolved


def _embedded_casts() -> dict[Path, list[dict[str, object]]]:
    embeds: dict[Path, list[dict[str, object]]] = {}
    for page in sorted(DOCS.rglob("*.md")):
        text = page.read_text(encoding="utf-8")
        for match in _DEMO_RE.finditer(text):
            attrs = _attrs(match.group("attrs"))
            cast_ref = attrs.get("data-fa-asciinema-cast")
            if cast_ref is None:
                continue
            asset = _resolve_docs_asset_reference(page, cast_ref)
            embeds.setdefault(asset, []).append(
                {
                    "page": str(page.relative_to(ROOT)),
                    "reference": cast_ref,
                    "cols": _optional_int(attrs.get("data-fa-asciinema-cols")),
                    "rows": _optional_int(attrs.get("data-fa-asciinema-rows")),
                    "idle_time_limit": _optional_float(
                        attrs.get("data-fa-asciinema-idle-time-limit")
                    ),
                    "poster": attrs.get("data-fa-asciinema-poster"),
                    "autoplay": attrs.get("data-fa-asciinema-autoplay") == "true",
                    "fit": attrs.get("data-fa-asciinema-fit"),
                }
            )
    return embeds


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _cast_header(path: Path) -> dict[str, object]:
    try:
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
    except IndexError:
        return {}
    data = json.loads(first_line)
    return data if isinstance(data, dict) else {}


def _cast_line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def _relative(path: Path) -> str:
    if not path.is_absolute():
        return str(path)
    return str(path.relative_to(ROOT))


def _asciinema_index() -> dict[str, object]:
    embeds = _embedded_casts()
    recorders = _cast_recorders()
    cast_paths = sorted(
        {path.relative_to(ROOT) for path in ASSETS.rglob("*.cast")} | set(embeds) | set(recorders)
    )
    casts: list[dict[str, object]] = []
    for path in cast_paths:
        abs_path = ROOT / path if not path.is_absolute() else path
        header = _cast_header(abs_path) if abs_path.exists() else {}
        recorder = recorders.get(path)
        embedded = embeds.get(path, [])
        problems: list[str] = []
        if not abs_path.exists():
            problems.append("missing asset")
        if recorder is None:
            problems.append("missing recorder")
        for embed in embedded:
            cols = embed.get("cols")
            rows = embed.get("rows")
            idle_time_limit = embed.get("idle_time_limit")
            if header and cols is not None and header.get("width") != cols:
                problems.append(f"{embed['page']} cols {cols} != cast width {header.get('width')}")
            if header and rows is not None and header.get("height") != rows:
                problems.append(
                    f"{embed['page']} rows {rows} != cast height {header.get('height')}"
                )
            if header and idle_time_limit is not None:
                header_idle = header.get("idle_time_limit")
                if header_idle is not None and float(header_idle) != float(idle_time_limit):
                    problems.append(
                        f"{embed['page']} idle {idle_time_limit} != cast idle {header_idle}"
                    )
        casts.append(
            {
                "path": _relative(path),
                "present": abs_path.exists(),
                "title": header.get("title"),
                "width": header.get("width"),
                "height": header.get("height"),
                "idle_time_limit": header.get("idle_time_limit"),
                "timestamp": header.get("timestamp"),
                "line_count": _cast_line_count(abs_path) if abs_path.exists() else 0,
                "recorder": recorder.name if recorder else None,
                "record_command": recorder.command if recorder else None,
                "notes": recorder.notes if recorder else "",
                "embedded": embedded,
                "problems": problems,
            }
        )
    return {
        "schema": 1,
        "description": "Internal review index for committed asciinema docs recordings.",
        "casts": casts,
    }


def _write_asciinema_index() -> None:
    ASCIINEMA_INDEX.write_text(
        json.dumps(_asciinema_index(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_asciinema_index() -> None:
    _write_asciinema_index()


def _asciinema_index_problems() -> list[str]:
    index = _asciinema_index()
    problems: list[str] = []
    for cast in index["casts"]:
        cast_problems = cast.get("problems", [])
        if isinstance(cast_problems, list):
            problems.extend(f"{cast['path']}: {problem}" for problem in cast_problems)
    expected = json.dumps(index, indent=2, sort_keys=True) + "\n"
    if not ASCIINEMA_INDEX.exists():
        problems.append(f"{ASCIINEMA_INDEX.relative_to(ROOT)} is missing")
    elif ASCIINEMA_INDEX.read_text(encoding="utf-8") != expected:
        problems.append(
            f"{ASCIINEMA_INDEX.relative_to(ROOT)} is stale; run `uv run scripts/docs.py assets`"
        )
    return problems


def asciinema_index_problems() -> list[str]:
    return _asciinema_index_problems()


def _model_picker_record_script(scenario: TerminalCastScenario) -> str:
    startup_wait = os.environ.get("FAST_AGENT_MODEL_PICKER_DEMO_STARTUP_WAIT", "5")
    navigation_wait = os.environ.get("FAST_AGENT_MODEL_PICKER_DEMO_NAVIGATION_WAIT", "0.55")
    final_wait = os.environ.get("FAST_AGENT_MODEL_PICKER_DEMO_FINAL_WAIT", "1.2")
    session = f"fast_agent_docs_{scenario.name.replace('-', '_')}"
    command = scenario.shell_command.replace("'", "'\"'\"'")
    return f"""#!/usr/bin/env bash
set -euo pipefail

SESSION='{session}'
ROOT='{ROOT}'

type_slow() {{
  local target="$1"
  local text="$2"
  local delay="$3"
  local i char
  for (( i=0; i<${{#text}}; i++ )); do
    char="${{text:i:1}}"
    tmux send-keys -l -t "$target" "$char"
    sleep "$delay"
  done
}}

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x {scenario.cols} -y {scenario.rows} \\
  "DEMO_FAST_AGENT_HOME=\\$(mktemp -d) && printf '{{}}\\n' > \\\"\\$DEMO_FAST_AGENT_HOME/fast-agent.yaml\\\" && export FAST_AGENT_HOME=\\\"\\$DEMO_FAST_AGENT_HOME\\\" && DEMO_WORKDIR=\\$(mktemp -d -t fast-agent-model-picker.XXXXXX) && cd \\\"\\$DEMO_WORKDIR\\\" && unset ENVIRONMENT_DIR FAST_AGENT_RUNTIME_ENVIRONMENT VIRTUAL_ENV FAST_AGENT_MODEL NO_COLOR && TERM=xterm-256color COLORTERM=truecolor FORCE_COLOR=1 FAST_AGENT_KEYRING_NOTICE=0 bash --noprofile --norc"
tmux set-option -t "$SESSION" status off >/dev/null

(
  sleep 1
  type_slow "$SESSION" '{command}' 0.035
  tmux send-keys -t "$SESSION" Enter
  sleep {startup_wait}
  tmux send-keys -t "$SESSION" Down
  sleep {navigation_wait}
  tmux send-keys -t "$SESSION" Down
  sleep {navigation_wait}
  tmux send-keys -t "$SESSION" Right
  sleep {navigation_wait}
  tmux send-keys -t "$SESSION" Down
  sleep {navigation_wait}
  tmux send-keys -t "$SESSION" Down
  sleep {final_wait}
  tmux kill-session -t "$SESSION" 2>/dev/null || true
) &

tmux attach-session -t "$SESSION" || true
"""


def _missing_tools(tools: tuple[str, ...]) -> list[str]:
    return [tool for tool in tools if shutil.which(tool) is None]


def require_recording_tools(tools: tuple[str, ...] = ("asciinema", "tmux")) -> None:
    missing = _missing_tools(tools)
    if missing:
        raise RuntimeError("Cannot record docs assets; missing tools: " + ", ".join(missing))


def _skills_direct_install_record_script(scenario: TerminalCastScenario) -> str:
    startup_wait = os.environ.get("FAST_AGENT_SKILLS_DEMO_STARTUP_WAIT", "0.7")
    command_wait = os.environ.get("FAST_AGENT_SKILLS_DEMO_COMMAND_WAIT", "1.4")
    update_wait = os.environ.get("FAST_AGENT_SKILLS_DEMO_UPDATE_WAIT", "0.6")
    final_wait = os.environ.get("FAST_AGENT_SKILLS_DEMO_FINAL_WAIT", "1.0")
    typing_delay = os.environ.get("FAST_AGENT_SKILLS_DEMO_TYPING_DELAY", "0.045")
    session = f"fast_agent_docs_{scenario.name.replace('-', '_')}"
    return f"""#!/usr/bin/env bash
set -euo pipefail

SESSION='{session}'
ROOT='{ROOT}'

type_slow() {{
  local target="$1"
  local text="$2"
  local delay="$3"
  local i char
  for (( i=0; i<${{#text}}; i++ )); do
    char="${{text:i:1}}"
    tmux send-keys -l -t "$target" "$char"
    sleep "$delay"
  done
}}

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x {scenario.cols} -y {scenario.rows} \\
  "DEMO_FAST_AGENT_HOME=\\$(mktemp -d) && export FAST_AGENT_HOME=\\\"\\$DEMO_FAST_AGENT_HOME\\\" && DEMO_WORKDIR=\\$(mktemp -d -t fast-agent-skills.XXXXXX) && cd \\\"\\$DEMO_WORKDIR\\\" && mkdir -p skill-repo/skills/demo-skill && cat > skill-repo/skills/demo-skill/SKILL.md <<'SKILL'
---
name: demo-skill
description: A small local skill installed from a local git repository.
---

# Demo Skill

Use this skill to demonstrate direct installation from a local repository.
SKILL
git -C skill-repo init -q && git -C skill-repo config user.email docs-demo@example.com && git -C skill-repo config user.name 'Docs Demo' && git -C skill-repo add . && git -C skill-repo commit -q -m 'Initial demo skill'
unset ENVIRONMENT_DIR FAST_AGENT_RUNTIME_ENVIRONMENT VIRTUAL_ENV NO_COLOR && TERM=xterm-256color COLORTERM=truecolor FORCE_COLOR=1 FAST_AGENT_KEYRING_NOTICE=0 bash --noprofile --norc"
tmux set-option -t "$SESSION" status off >/dev/null

(
  sleep {startup_wait}
  type_slow "$SESSION" 'fast-agent skills add ./skill-repo/skills/demo-skill' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  sleep {command_wait}
  type_slow "$SESSION" 'fast-agent skills update' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  sleep {command_wait}
  type_slow "$SESSION" 'cat >> skill-repo/skills/demo-skill/SKILL.md' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  sleep 0.2
  type_slow "$SESSION" '## Updated' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  type_slow "$SESSION" 'A new section from the local repo.' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  tmux send-keys -t "$SESSION" C-d
  sleep {update_wait}
  type_slow "$SESSION" 'git -C skill-repo add . && git -C skill-repo commit -m update' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  sleep {update_wait}
  type_slow "$SESSION" 'fast-agent skills update' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  sleep {command_wait}
  sleep {final_wait}
  tmux kill-session -t "$SESSION" 2>/dev/null || true
) &

tmux attach-session -t "$SESSION" || true
"""


def _skills_slash_commands_record_script(scenario: TerminalCastScenario) -> str:
    startup_wait = os.environ.get("FAST_AGENT_SKILLS_SLASH_DEMO_STARTUP_WAIT", "3.5")
    command_wait = os.environ.get("FAST_AGENT_SKILLS_SLASH_DEMO_COMMAND_WAIT", "1.4")
    update_wait = os.environ.get("FAST_AGENT_SKILLS_SLASH_DEMO_UPDATE_WAIT", "0.6")
    final_wait = os.environ.get("FAST_AGENT_SKILLS_SLASH_DEMO_FINAL_WAIT", "1.0")
    typing_delay = os.environ.get("FAST_AGENT_SKILLS_SLASH_DEMO_TYPING_DELAY", "0.035")
    session = f"fast_agent_docs_{scenario.name.replace('-', '_')}"
    command = scenario.shell_command.replace("'", "'\"'\"'")
    return f"""#!/usr/bin/env bash
set -euo pipefail

SESSION='{session}'
ROOT='{ROOT}'

type_slow() {{
  local target="$1"
  local text="$2"
  local delay="$3"
  local i char
  for (( i=0; i<${{#text}}; i++ )); do
    char="${{text:i:1}}"
    tmux send-keys -l -t "$target" "$char"
    sleep "$delay"
  done
}}

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x {scenario.cols} -y {scenario.rows} \\
  "DEMO_FAST_AGENT_HOME=\\$(mktemp -d) && printf '{{}}\\n' > \\\"\\$DEMO_FAST_AGENT_HOME/fast-agent.yaml\\\" && export FAST_AGENT_HOME=\\\"\\$DEMO_FAST_AGENT_HOME\\\" && DEMO_WORKDIR=\\$(mktemp -d -t fast-agent-skills-slash.XXXXXX) && cd \\\"\\$DEMO_WORKDIR\\\" && mkdir -p skill-repo/skills/demo-skill && cat > skill-repo/skills/demo-skill/SKILL.md <<'SKILL'
---
name: demo-skill
description: A small local skill installed from the TUI.
---

# Demo Skill

Use this skill to demonstrate /skills add and /skills update.
SKILL
git -C skill-repo init -q && git -C skill-repo config user.email docs-demo@example.com && git -C skill-repo config user.name 'Docs Demo' && git -C skill-repo add . && git -C skill-repo commit -q -m 'Initial demo skill'
unset ENVIRONMENT_DIR FAST_AGENT_RUNTIME_ENVIRONMENT VIRTUAL_ENV NO_COLOR && TERM=xterm-256color COLORTERM=truecolor FORCE_COLOR=1 FAST_AGENT_KEYRING_NOTICE=0 TUI__COMPLETION_MENU_RESERVED_LINES=${{TUI__COMPLETION_MENU_RESERVED_LINES:-4}} bash --noprofile --norc"
tmux set-option -t "$SESSION" status off >/dev/null

(
  sleep 1
  type_slow "$SESSION" '{command}' 0.035
  tmux send-keys -t "$SESSION" Enter
  sleep {startup_wait}
  type_slow "$SESSION" '/skills add ./skill-repo/skills/demo-skill' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  sleep {command_wait}
  type_slow "$SESSION" '! cat >> skill-repo/skills/demo-skill/SKILL.md' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  sleep 0.2
  type_slow "$SESSION" '## Updated' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  type_slow "$SESSION" 'A new section from the local repo.' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  tmux send-keys -t "$SESSION" C-d
  sleep {update_wait}
  type_slow "$SESSION" '! git -C skill-repo add . && git -C skill-repo commit -m update' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  sleep {update_wait}
  type_slow "$SESSION" '/skills update' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  sleep {command_wait}
  sleep {final_wait}
  tmux kill-session -t "$SESSION" 2>/dev/null || true
) &

tmux attach-session -t "$SESSION" || true
"""


def _skills_over_mcp_record_script(scenario: TerminalCastScenario) -> str:
    startup_wait = os.environ.get("FAST_AGENT_SKILLS_MCP_DEMO_STARTUP_WAIT", "3.5")
    connect_wait = os.environ.get("FAST_AGENT_SKILLS_MCP_DEMO_CONNECT_WAIT", "6.0")
    command_wait = os.environ.get("FAST_AGENT_SKILLS_MCP_DEMO_COMMAND_WAIT", "1.4")
    final_wait = os.environ.get("FAST_AGENT_SKILLS_MCP_DEMO_FINAL_WAIT", "1.0")
    typing_delay = os.environ.get("FAST_AGENT_SKILLS_MCP_DEMO_TYPING_DELAY", "0.035")
    server_url = os.environ.get(
        "FAST_AGENT_SKILLS_MCP_DEMO_SERVER",
        "https://huggingface.co/mcp",
    )
    session = f"fast_agent_docs_{scenario.name.replace('-', '_')}"
    command = scenario.shell_command.replace("'", "'\"'\"'")
    server_url = server_url.replace("'", "'\"'\"'")
    return f"""#!/usr/bin/env bash
set -euo pipefail

SESSION='{session}'
ROOT='{ROOT}'

type_slow() {{
  local target="$1"
  local text="$2"
  local delay="$3"
  local i char
  for (( i=0; i<${{#text}}; i++ )); do
    char="${{text:i:1}}"
    tmux send-keys -l -t "$target" "$char"
    sleep "$delay"
  done
}}

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x {scenario.cols} -y {scenario.rows} \\
  "DEMO_FAST_AGENT_HOME=\\$(mktemp -d) && printf '{{}}\\n' > \\\"\\$DEMO_FAST_AGENT_HOME/fast-agent.yaml\\\" && export FAST_AGENT_HOME=\\\"\\$DEMO_FAST_AGENT_HOME\\\" && DEMO_WORKDIR=\\$(mktemp -d -t fast-agent-skills-mcp.XXXXXX) && cd \\\"\\$DEMO_WORKDIR\\\" && unset ENVIRONMENT_DIR FAST_AGENT_RUNTIME_ENVIRONMENT VIRTUAL_ENV NO_COLOR && TERM=xterm-256color COLORTERM=truecolor FORCE_COLOR=1 FAST_AGENT_KEYRING_NOTICE=0 TUI__COMPLETION_MENU_RESERVED_LINES=${{TUI__COMPLETION_MENU_RESERVED_LINES:-4}} bash --noprofile --norc"
tmux set-option -t "$SESSION" status off >/dev/null

(
  sleep 1
  type_slow "$SESSION" '{command}' 0.035
  tmux send-keys -t "$SESSION" Enter
  sleep {startup_wait}
  type_slow "$SESSION" '/mcp connect {server_url} --name hf' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  sleep {connect_wait}
  type_slow "$SESSION" '/mcp' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  sleep {command_wait}
  type_slow "$SESSION" '/skills registry' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  sleep {command_wait}
  type_slow "$SESSION" '/skills registry hf' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  sleep {command_wait}
  type_slow "$SESSION" '/skills search dataset viewer' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  sleep {command_wait}
  type_slow "$SESSION" '/skills add huggingface-datasets' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  sleep {command_wait}
  type_slow "$SESSION" '/skills' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  sleep {command_wait}
  sleep {final_wait}
  tmux kill-session -t "$SESSION" 2>/dev/null || true
) &

tmux attach-session -t "$SESSION" || true
"""


def _mcp_tool_schema_record_script(scenario: TerminalCastScenario) -> str:
    startup_timeout = os.environ.get("FAST_AGENT_TOOL_SCHEMA_DEMO_STARTUP_TIMEOUT", "30")
    connect_timeout = os.environ.get("FAST_AGENT_TOOL_SCHEMA_DEMO_CONNECT_TIMEOUT", "30")
    schema_timeout = os.environ.get("FAST_AGENT_TOOL_SCHEMA_DEMO_SCHEMA_TIMEOUT", "30")
    prompt_timeout = os.environ.get("FAST_AGENT_TOOL_SCHEMA_DEMO_PROMPT_TIMEOUT", "30")
    input_schema_wait = os.environ.get("FAST_AGENT_TOOL_SCHEMA_DEMO_INPUT_WAIT", "3.0")
    output_schema_wait = os.environ.get("FAST_AGENT_TOOL_SCHEMA_DEMO_OUTPUT_WAIT", "3.0")
    output_page_wait = os.environ.get("FAST_AGENT_TOOL_SCHEMA_DEMO_PAGE_WAIT", "2.5")
    final_wait = os.environ.get("FAST_AGENT_TOOL_SCHEMA_DEMO_FINAL_WAIT", "3.0")
    typing_delay = os.environ.get("FAST_AGENT_TOOL_SCHEMA_DEMO_TYPING_DELAY", "0.035")
    server_url = os.environ.get(
        "FAST_AGENT_TOOL_SCHEMA_DEMO_SERVER",
        "http://localhost:3000/mcp",
    )
    session = f"fast_agent_docs_{scenario.name.replace('-', '_')}"
    command = scenario.shell_command.replace("'", "'\"'\"'")
    server_url = server_url.replace("'", "'\"'\"'")
    return f"""#!/usr/bin/env bash
set -euo pipefail

SESSION='{session}'
ROOT='{ROOT}'

type_slow() {{
  local target="$1"
  local text="$2"
  local delay="$3"
  local i char
  for (( i=0; i<${{#text}}; i++ )); do
    char="${{text:i:1}}"
    tmux send-keys -l -t "$target" "$char"
    sleep "$delay"
  done
}}

wait_for_pane() {{
  local target="$1"
  local pattern="$2"
  local timeout="$3"
  local attempts=$((timeout * 4))
  local i
  for (( i=0; i<attempts; i++ )); do
    if tmux capture-pane -p -t "$target" -S -5000 | grep -Fq "$pattern"; then
      return 0
    fi
    sleep 0.25
  done
  tmux capture-pane -p -t "$target" -S -5000 >&2 || true
  return 1
}}

wait_for_prompt() {{
  local target="$1"
  local timeout="$2"
  local attempts=$((timeout * 4))
  local i
  for (( i=0; i<attempts; i++ )); do
    if tmux capture-pane -p -t "$target" -S -8 | grep -Eq '^❯[[:space:]]*$'; then
      return 0
    fi
    sleep 0.25
  done
  tmux capture-pane -p -t "$target" -S -50 >&2 || true
  return 1
}}

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x {scenario.cols} -y {scenario.rows} \\
  "DEMO_FAST_AGENT_HOME=\\$(mktemp -d) && printf '{{}}\\n' > \\\"\\$DEMO_FAST_AGENT_HOME/fast-agent.yaml\\\" && export FAST_AGENT_HOME=\\\"\\$DEMO_FAST_AGENT_HOME\\\" && DEMO_WORKDIR=\\$(mktemp -d -t fast-agent-tool-schema.XXXXXX) && cd \\\"\\$DEMO_WORKDIR\\\" && unset ENVIRONMENT_DIR FAST_AGENT_RUNTIME_ENVIRONMENT VIRTUAL_ENV NO_COLOR && TERM=xterm-256color COLORTERM=truecolor FORCE_COLOR=1 FAST_AGENT_KEYRING_NOTICE=0 TUI__COMPLETION_MENU_RESERVED_LINES=${{TUI__COMPLETION_MENU_RESERVED_LINES:-4}} bash --noprofile --norc"
tmux set-option -t "$SESSION" status off >/dev/null

(
  trap 'tmux kill-session -t "$SESSION" 2>/dev/null || true' EXIT
  sleep 1
  type_slow "$SESSION" '{command}' 0.035
  tmux send-keys -t "$SESSION" Enter
  wait_for_prompt "$SESSION" {startup_timeout}
  type_slow "$SESSION" '/mcp connect {server_url} --name hf' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  wait_for_pane "$SESSION" "Connected MCP server 'hf'" {connect_timeout}
  wait_for_prompt "$SESSION" {prompt_timeout}
  type_slow "$SESSION" '/tool hf__hf_whoami' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  wait_for_pane "$SESSION" 'Structured output schema' {schema_timeout}
  wait_for_prompt "$SESSION" {prompt_timeout}
  tmux copy-mode -t "$SESSION"
  tmux send-keys -X -t "$SESSION" search-backward 'Input schema'
  sleep {input_schema_wait}
  tmux send-keys -X -t "$SESSION" search-forward 'Structured output schema'
  sleep {output_schema_wait}
  tmux send-keys -X -t "$SESSION" page-down
  sleep {output_page_wait}
  tmux send-keys -X -t "$SESSION" search-backward 'Structured output schema'
  sleep {final_wait}
) &
DRIVER_PID=$!

tmux attach-session -t "$SESSION" || true
wait "$DRIVER_PID"
"""


def _hf_image_generation_record_script(scenario: TerminalCastScenario) -> str:
    startup_wait = os.environ.get("FAST_AGENT_HF_IMAGE_DEMO_STARTUP_WAIT", "8")
    connect_timeout = os.environ.get("FAST_AGENT_HF_IMAGE_DEMO_CONNECT_TIMEOUT", "60")
    whoami_timeout = os.environ.get("FAST_AGENT_HF_IMAGE_DEMO_WHOAMI_TIMEOUT", "60")
    response_timeout = os.environ.get("FAST_AGENT_HF_IMAGE_DEMO_RESPONSE_TIMEOUT", "180")
    prompt_timeout = os.environ.get("FAST_AGENT_HF_IMAGE_DEMO_PROMPT_TIMEOUT", "60")
    status_wait = os.environ.get("FAST_AGENT_HF_IMAGE_DEMO_STATUS_WAIT", "2.5")
    final_wait = os.environ.get("FAST_AGENT_HF_IMAGE_DEMO_FINAL_WAIT", "1")
    typing_delay = os.environ.get("FAST_AGENT_HF_IMAGE_DEMO_TYPING_DELAY", "0.035")
    session = f"fast_agent_docs_{scenario.name.replace('-', '_')}"
    command = scenario.shell_command.replace("'", "'\"'\"'")
    prompt = scenario.prompt.replace("'", "'\"'\"'")
    whoami_prompt = os.environ.get(
        "FAST_AGENT_HF_IMAGE_DEMO_WHOAMI_PROMPT",
        (
            "Identify my authenticated Hugging Face username. "
            "Reply with only the username and no other account details."
        ),
    ).replace("'", "'\"'\"'")
    server_url = os.environ.get(
        "FAST_AGENT_HF_IMAGE_DEMO_SERVER",
        "https://huggingface.co/mcp",
    ).replace("'", "'\"'\"'")
    return f"""#!/usr/bin/env bash
set -euo pipefail

SESSION='{session}'
ROOT='{ROOT}'
HF_TOKEN="${{HF_TOKEN:-}}"
if [[ -z "$HF_TOKEN" ]]; then
  HF_TOKEN="$(uv run python -c 'from huggingface_hub import get_token; print(get_token() or "")')"
fi
if [[ -z "$HF_TOKEN" ]]; then
  echo "Set HF_TOKEN or log in with huggingface_hub before recording hf-image-generation." >&2
  exit 2
fi

type_slow() {{
  local target="$1"
  local text="$2"
  local delay="$3"
  local i char
  for (( i=0; i<${{#text}}; i++ )); do
    char="${{text:i:1}}"
    tmux send-keys -l -t "$target" "$char"
    sleep "$delay"
  done
}}

wait_for_pane() {{
  local target="$1"
  local pattern="$2"
  local timeout="$3"
  local attempts=$((timeout * 4))
  local i
  for (( i=0; i<attempts; i++ )); do
    if tmux capture-pane -p -t "$target" -S -5000 | grep -Fq "$pattern"; then
      return 0
    fi
    sleep 0.25
  done
  tmux capture-pane -p -t "$target" -S -5000 >&2 || true
  return 1
}}

wait_for_prompt() {{
  local target="$1"
  local timeout="$2"
  local attempts=$((timeout * 4))
  local i
  for (( i=0; i<attempts; i++ )); do
    if tmux capture-pane -p -t "$target" -S -8 | grep -Eq '^❯[[:space:]]*$'; then
      return 0
    fi
    sleep 0.25
  done
  tmux capture-pane -p -t "$target" -S -50 >&2 || true
  return 1
}}

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -e "HF_TOKEN=$HF_TOKEN" -s "$SESSION" -x {scenario.cols} -y {scenario.rows} \\
  "DEMO_FAST_AGENT_HOME=\\$(mktemp -d) && printf 'mcp:\\n  diagnostics:\\n    enabled: true\\n    timeline:\\n      steps: 60\\n      step_seconds: 1\\n' > \\\"\\$DEMO_FAST_AGENT_HOME/fast-agent.yaml\\\" && export FAST_AGENT_HOME=\\\"\\$DEMO_FAST_AGENT_HOME\\\" && DEMO_WORKDIR=\\$(mktemp -d -t fast-agent-hf-image.XXXXXX) && cd \\\"\\$DEMO_WORKDIR\\\" && unset ENVIRONMENT_DIR FAST_AGENT_RUNTIME_ENVIRONMENT VIRTUAL_ENV NO_COLOR && TERM=xterm-256color COLORTERM=truecolor FORCE_COLOR=1 FAST_AGENT_KEYRING_NOTICE=0 TUI__COMPLETION_MENU_RESERVED_LINES=${{TUI__COMPLETION_MENU_RESERVED_LINES:-4}} LOGGER__TERMINAL_IMAGES__ENABLED=true LOGGER__TERMINAL_IMAGES__BACKEND=halfcell LOGGER__TERMINAL_IMAGES__WIDTH=${{LOGGER__TERMINAL_IMAGES__WIDTH:-96}} LOGGER__TERMINAL_IMAGES__HEIGHT=${{LOGGER__TERMINAL_IMAGES__HEIGHT:-24}} bash --noprofile --norc"
tmux set-option -t "$SESSION" status off >/dev/null

(
  sleep 1
  type_slow "$SESSION" '{command}' 0.035
  tmux send-keys -t "$SESSION" Enter
  sleep {startup_wait}
  type_slow "$SESSION" '/mcp connect {server_url} --name hf --auth $HF_TOKEN --no-oauth' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  wait_for_pane "$SESSION" "Connected MCP server 'hf'" {connect_timeout}
  wait_for_prompt "$SESSION" {prompt_timeout}
  type_slow "$SESSION" '{whoami_prompt}' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  wait_for_pane "$SESSION" 'hf_whoami' {whoami_timeout}
  wait_for_prompt "$SESSION" {prompt_timeout}
  type_slow "$SESSION" '{prompt}' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  wait_for_pane "$SESSION" '[IMAGE 1:' {response_timeout}
  wait_for_prompt "$SESSION" {prompt_timeout}
  type_slow "$SESSION" '/mcp' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  sleep {status_wait}
  sleep {final_wait}
  tmux kill-session -t "$SESSION" 2>/dev/null || true
) &

tmux attach-session -t "$SESSION" || true
"""


def _mcp_inspect_legacy_record_script(scenario: TerminalCastScenario) -> str:
    startup_wait = os.environ.get("FAST_AGENT_MCP_LEGACY_DEMO_STARTUP_WAIT", "4")
    attach_timeout = os.environ.get("FAST_AGENT_MCP_LEGACY_DEMO_ATTACH_TIMEOUT", "30")
    prompt_timeout = os.environ.get("FAST_AGENT_MCP_LEGACY_DEMO_PROMPT_TIMEOUT", "30")
    status_timeout = os.environ.get("FAST_AGENT_MCP_LEGACY_DEMO_STATUS_TIMEOUT", "30")
    ping_wait = os.environ.get("FAST_AGENT_MCP_LEGACY_DEMO_PING_WAIT", "3")
    status_wait = os.environ.get("FAST_AGENT_MCP_LEGACY_DEMO_STATUS_WAIT", "2")
    typing_delay = os.environ.get("FAST_AGENT_MCP_LEGACY_DEMO_TYPING_DELAY", "0.035")
    session = f"fast_agent_docs_{scenario.name.replace('-', '_')}"
    command = scenario.shell_command.replace("'", "'\"'\"'")
    return f"""#!/usr/bin/env bash
set -euo pipefail

SESSION='{session}'
ROOT='{ROOT}'
PORT="${{FAST_AGENT_MCP_LEGACY_DEMO_PORT:-43177}}"
SERVER_LOG="$(mktemp -t fast-agent-mcp-legacy.XXXXXX.log)"
DEMO_FAST_AGENT_HOME="$(mktemp -d)"
DEMO_WORKDIR="$(mktemp -d -t fast-agent-mcp-legacy.XXXXXX)"

cleanup() {{
  tmux kill-session -t "$SESSION" 2>/dev/null || true
  if [[ -n "${{SERVER_PID:-}}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "$DEMO_FAST_AGENT_HOME" "$DEMO_WORKDIR"
  rm -f "$SERVER_LOG"
}}
trap cleanup EXIT

type_slow() {{
  local target="$1"
  local text="$2"
  local delay="$3"
  local i char
  for (( i=0; i<${{#text}}; i++ )); do
    char="${{text:i:1}}"
    tmux send-keys -l -t "$target" "$char"
    sleep "$delay"
  done
}}

wait_for_pane() {{
  local target="$1"
  local pattern="$2"
  local timeout="$3"
  local attempts=$((timeout * 4))
  local i
  for (( i=0; i<attempts; i++ )); do
    if tmux capture-pane -p -t "$target" -S -5000 | grep -Fq "$pattern"; then
      return 0
    fi
    sleep 0.25
  done
  tmux capture-pane -p -t "$target" -S -5000 >&2 || true
  return 1
}}

wait_for_prompt() {{
  local target="$1"
  local timeout="$2"
  local attempts=$((timeout * 4))
  local i
  for (( i=0; i<attempts; i++ )); do
    if tmux capture-pane -p -t "$target" -S -8 | grep -Eq '^❯[[:space:]]*$'; then
      return 0
    fi
    sleep 0.25
  done
  tmux capture-pane -p -t "$target" -S -50 >&2 || true
  return 1
}}

cat > "$DEMO_FAST_AGENT_HOME/fast-agent.yaml" <<YAML
mcp:
  diagnostics:
    enabled: true
    timeline:
      steps: 60
      step_seconds: 1
  servers:
    legacy_remote:
      transport: http
      url: http://127.0.0.1:$PORT/mcp
      protocol_mode: legacy
      ping_interval_seconds: 1
      max_missed_pings: 3
      load_on_start: false
YAML

cd "$ROOT"
uv run python scripts/docs_mcp_legacy_server.py --port "$PORT" >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
for _ in {{1..100}}; do
  if curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    cat "$SERVER_LOG" >&2
    exit 1
  fi
  sleep 0.1
done
curl -fsS "http://127.0.0.1:$PORT/healthz" >/dev/null

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x {scenario.cols} -y {scenario.rows} \\
  "cd '$DEMO_WORKDIR' && export FAST_AGENT_HOME='$DEMO_FAST_AGENT_HOME' && unset ENVIRONMENT_DIR FAST_AGENT_RUNTIME_ENVIRONMENT VIRTUAL_ENV NO_COLOR && PYTHONWARNINGS=ignore::UserWarning TERM=xterm-256color COLORTERM=truecolor FORCE_COLOR=1 FAST_AGENT_KEYRING_NOTICE=0 TUI__COMPLETION_MENU_RESERVED_LINES=${{TUI__COMPLETION_MENU_RESERVED_LINES:-4}} bash --noprofile --norc"
tmux set-option -t "$SESSION" status off >/dev/null

(
  sleep 1
  type_slow "$SESSION" '{command}' 0.035
  tmux send-keys -t "$SESSION" Enter
  sleep {startup_wait}
  wait_for_prompt "$SESSION" {prompt_timeout}
  type_slow "$SESSION" '/mcp attach legacy_remote' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  wait_for_pane "$SESSION" "Attached configured MCP server 'legacy_remote'" {attach_timeout}
  wait_for_prompt "$SESSION" {prompt_timeout}
  sleep {ping_wait}
  type_slow "$SESSION" '/mcp' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  wait_for_pane "$SESSION" 'Docs Legacy Remote' {status_timeout}
  sleep {status_wait}
  tmux kill-session -t "$SESSION" 2>/dev/null || true
) &

tmux attach-session -t "$SESSION" || true
"""


def _elicitation_sandbox_record_script(scenario: TerminalCastScenario) -> str:
    startup_wait = os.environ.get("FAST_AGENT_ELICITATION_DEMO_STARTUP_WAIT", "4")
    field_wait = os.environ.get("FAST_AGENT_ELICITATION_DEMO_FIELD_WAIT", "0.45")
    final_wait = os.environ.get("FAST_AGENT_ELICITATION_DEMO_FINAL_WAIT", "2")
    typing_delay = os.environ.get("FAST_AGENT_ELICITATION_DEMO_TYPING_DELAY", "0.045")
    session = f"fast_agent_docs_{scenario.name.replace('-', '_')}"
    command = scenario.shell_command.replace("'", "'\"'\"'")
    workdir = ROOT / "examples" / "mcp" / "elicitations"
    return f"""#!/usr/bin/env bash
set -euo pipefail

SESSION='{session}'

type_slow() {{
  local target="$1"
  local text="$2"
  local delay="$3"
  local i char
  for (( i=0; i<${{#text}}; i++ )); do
    char="${{text:i:1}}"
    tmux send-keys -l -t "$target" "$char"
    sleep "$delay"
  done
}}

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x {scenario.cols} -y {scenario.rows} \\
  "cd '{workdir}' && unset ENVIRONMENT_DIR FAST_AGENT_HOME FAST_AGENT_RUNTIME_HOME FAST_AGENT_RUNTIME_ENVIRONMENT VIRTUAL_ENV NO_COLOR && PS1='' TERM=xterm-256color COLORTERM=truecolor FORCE_COLOR=1 bash --noprofile --norc"
tmux set-option -t "$SESSION" status off >/dev/null

(
  sleep 1
  type_slow "$SESSION" '{command}' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  sleep {startup_wait}
  tmux send-keys -t "$SESSION" Down
  sleep {field_wait}
  tmux send-keys -t "$SESSION" Down Space
  sleep {field_wait}
  tmux send-keys -t "$SESSION" Tab Home Delete 2
  sleep {field_wait}
  tmux send-keys -t "$SESSION" Tab Home Delete Delete Delete
  type_slow "$SESSION" '1.00' {typing_delay}
  sleep {field_wait}
  tmux send-keys -t "$SESSION" Tab Enter
  sleep {final_wait}
  tmux kill-session -t "$SESSION" 2>/dev/null || true
) &

tmux attach-session -t "$SESSION" || true
"""


def _required_assets() -> list[Path]:
    return [
        VENDOR_ASCIINEMA / "README.md",
        VENDOR_ASCIINEMA / "asciinema-player.css",
        VENDOR_ASCIINEMA / "asciinema-player.min.js",
        VENDOR_ASCIINEMA / "catppuccin.css",
    ]


def list_assets() -> int:
    print("Terminal cast recordings:")
    for cast in _asciinema_index()["casts"]:
        embedded_count = len(cast["embedded"]) if isinstance(cast["embedded"], list) else 0
        recorder = cast["recorder"] or "missing-recorder"
        status = "present" if cast["present"] else "missing"
        print(f"  {recorder:<32} {status:<7} embeds={embedded_count:<2} {cast['path']}")
    print(f"\nIndex: {ASCIINEMA_INDEX.relative_to(ROOT)}")
    return 0


def check() -> int:
    missing = [path for path in _required_assets() if not path.exists()]
    if missing:
        print("Missing docs assets:")
        for path in missing:
            print(f"  - {path.relative_to(ROOT)}")
        return 1

    problems = _asciinema_index_problems()
    if problems:
        print("Asciinema docs asset issues:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("Docs asset support files and asciinema index are current.")
    for cast in _asciinema_index()["casts"]:
        print(f"  {cast['recorder']:<32} {cast['path']}")
    return 0


def build() -> int:
    """Build static assets that do not require external services."""
    _write_asciinema_index()
    return check()


def _record_script(scenario: TerminalCastScenario) -> str:
    if scenario.name == "model-picker":
        return _model_picker_record_script(scenario)
    if scenario.name == "skills-direct-install":
        return _skills_direct_install_record_script(scenario)
    if scenario.name == "skills-slash-commands":
        return _skills_slash_commands_record_script(scenario)
    if scenario.name == "skills-over-mcp":
        return _skills_over_mcp_record_script(scenario)
    if scenario.name == "mcp-tool-schema":
        return _mcp_tool_schema_record_script(scenario)
    if scenario.name == "hf-image-generation":
        return _hf_image_generation_record_script(scenario)
    if scenario.name == "mcp-inspect-legacy":
        return _mcp_inspect_legacy_record_script(scenario)
    if scenario.name == "elicitation-sandbox":
        return _elicitation_sandbox_record_script(scenario)

    typing_delay = os.environ.get("FAST_AGENT_TUI_DEMO_TYPING_DELAY", "0.055")
    shell_delay = os.environ.get("FAST_AGENT_TUI_DEMO_SHELL_TYPING_DELAY", "0.045")
    startup_wait = os.environ.get("FAST_AGENT_TUI_DEMO_STARTUP_WAIT", "8")
    response_wait = os.environ.get("FAST_AGENT_TUI_DEMO_RESPONSE_WAIT", "14")
    shell_wait = os.environ.get("FAST_AGENT_TUI_DEMO_SHELL_WAIT", "5")
    show_exit = os.environ.get("FAST_AGENT_TUI_DEMO_SHOW_EXIT", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    session = f"fast_agent_docs_{scenario.name.replace('-', '_')}"
    prompt = scenario.prompt.replace("'", "'\"'\"'")
    command = scenario.shell_command.replace("'", "'\"'\"'")
    exit_block = (
        """
  type_slow "$SESSION" '/exit' 0.035
  tmux send-keys -t "$SESSION" Enter
  sleep 1"""
        if show_exit
        else """
  sleep 1"""
    )
    return f"""#!/usr/bin/env bash
set -euo pipefail

SESSION='{session}'
ROOT='{ROOT}'

type_slow() {{
  local target="$1"
  local text="$2"
  local delay="$3"
  local i char
  for (( i=0; i<${{#text}}; i++ )); do
    char="${{text:i:1}}"
    tmux send-keys -l -t "$target" "$char"
    sleep "$delay"
  done
}}

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -x {scenario.cols} -y {scenario.rows} \\
  "DEMO_FAST_AGENT_HOME=\\$(mktemp -d) && printf '{{}}\\n' > \\\"\\$DEMO_FAST_AGENT_HOME/fast-agent.yaml\\\" && export FAST_AGENT_HOME=\\\"\\$DEMO_FAST_AGENT_HOME\\\" && DEMO_WORKDIR=\\$(mktemp -d -t fast-agent-demo.XXXXXX) && cd \\\"\\$DEMO_WORKDIR\\\" && git init -q && git config user.email docs@example.invalid && git config user.name 'Docs Demo' && printf '# Demo workspace\\n' > README.md && git add README.md && git commit -qm init && printf '\\nLocal edit\\n' >> README.md && unset ENVIRONMENT_DIR FAST_AGENT_RUNTIME_ENVIRONMENT VIRTUAL_ENV NO_COLOR && TERM=xterm-256color COLORTERM=truecolor FORCE_COLOR=1 FAST_AGENT_KEYRING_NOTICE=0 TUI__COMPLETION_MENU_RESERVED_LINES=${{TUI__COMPLETION_MENU_RESERVED_LINES:-4}} bash --noprofile --norc"
tmux set-option -t "$SESSION" status off >/dev/null

(
  sleep 1
  type_slow "$SESSION" '{command}' 0.035
  tmux send-keys -t "$SESSION" Enter
  sleep {startup_wait}
  type_slow "$SESSION" '{prompt}' {typing_delay}
  tmux send-keys -t "$SESSION" Enter
  sleep {response_wait}
  type_slow "$SESSION" '! git status' {shell_delay}
  tmux send-keys -t "$SESSION" Enter
  sleep {shell_wait}
{exit_block}
  tmux kill-session -t "$SESSION" 2>/dev/null || true
) &

tmux attach-session -t "$SESSION" || true
"""


def record(name: str) -> int:
    scenarios = _scenarios()
    scenario = scenarios.get(name)
    if scenario is None:
        print(f"Unknown docs asset scenario: {name}")
        print("Available scenarios: " + ", ".join(sorted(scenarios)))
        return 1

    try:
        require_recording_tools()
    except RuntimeError as exc:
        print(str(exc))
        return 1

    with tempfile.TemporaryDirectory(prefix="fast-agent-docs-assets-") as temp_dir:
        driver = Path(temp_dir) / f"{scenario.name}.sh"
        driver.write_text(_record_script(scenario), encoding="utf-8")
        driver.chmod(0o755)
        record_asciinema_cast(
            output=scenario.output,
            title=scenario.title,
            command=str(driver),
            cols=scenario.cols,
            rows=scenario.rows,
            idle_time_limit=scenario.idle_time_limit,
            cleanup_session=f"fast_agent_docs_{name.replace('-', '_')}",
        )
    print(f"Recorded {scenario.output.relative_to(ROOT)}")
    return 0


def record_asciinema_cast(
    *,
    output: Path,
    title: str,
    command: str,
    cols: int,
    rows: int,
    idle_time_limit: float = 1.3,
    cleanup_session: str | None = None,
    cwd: Path = ROOT,
) -> None:
    """Record an asciinema cast using docs-wide defaults and cleanup rules."""
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "asciinema",
                "rec",
                "--overwrite",
                "--cols",
                str(cols),
                "--rows",
                str(rows),
                "--idle-time-limit",
                str(idle_time_limit),
                "-t",
                title,
                "-c",
                command,
                str(output),
            ],
            cwd=cwd,
            check=True,
        )
    finally:
        if cleanup_session:
            subprocess.run(["tmux", "kill-session", "-t", cleanup_session], check=False)
    _trim_terminal_teardown(output)


def _is_terminal_teardown_event(line: str) -> bool:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return False
    if not isinstance(event, list) or len(event) < 3:
        return False
    if event[1] != "o" or not isinstance(event[2], str):
        return False
    output = event[2]
    return (
        "[exited]" in output
        or "[detached" in output
        or "\x1b[?1049l" in output
        or "\u001b[?1049l" in output
        or "\x1b[H\x1b[2J" in output
        or "\u001b[H\u001b[2J" in output
    )


def _trim_terminal_teardown(path: Path) -> None:
    """Remove tmux/asciinema teardown frames so the cast ends on the demo content."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= 1:
        return
    header, events = lines[0], lines[1:]
    trimmed = list(events)
    while trimmed and _is_terminal_teardown_event(trimmed[-1]):
        trimmed.pop()
    if len(trimmed) == len(events):
        return
    path.write_text("\n".join([header, *trimmed]) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list")
    subparsers.add_parser("check")
    subparsers.add_parser("build")
    subparsers.add_parser("index")
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("scenario", choices=sorted(_scenarios()))
    args = parser.parse_args()

    if args.command == "list":
        return list_assets()
    if args.command == "check":
        return check()
    if args.command == "build":
        return build()
    if args.command == "index":
        _write_asciinema_index()
        print(f"Wrote {ASCIINEMA_INDEX.relative_to(ROOT)}")
        return 0
    if args.command == "record":
        return record(args.scenario)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
