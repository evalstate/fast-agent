"""Live smoke test: OpenAI-format provider-native compaction (``native_compaction``).

Routes and their credentials:

- ``codexresponses`` (default): ChatGPT login, ``fast-agent auth provider login codex``
- ``copilot``: GitHub Copilot login, ``fast-agent auth provider login copilot``
- ``responses``: an OpenAI API key (``OPENAI_API_KEY``)
- ``xai``: xAI login or key (``fast-agent auth provider login xai``); compacts through
  ``POST /responses/compact`` rather than a ``compaction_trigger`` item

    uv run scripts/openai_native_compaction_smoke.py                  # codexresponses, both
    uv run scripts/openai_native_compaction_smoke.py --route copilot --model gpt-6-sol
    uv run scripts/openai_native_compaction_smoke.py --route xai --model grok-4.7
    uv run scripts/openai_native_compaction_smoke.py --transport sse
    uv run scripts/openai_native_compaction_smoke.py --fixture-out /tmp/codex-compaction-trace

For each transport, in a fresh workspace using the default (native compaction on):

1. A tool returns a large file. The context crosses a low threshold, so the
   after-turn hook compacts the turn into an opaque provider checkpoint.
2. The reasoning effort changes, then a follow-up recalls the codeword. Its input
   is a small fraction of the tool request's, and the wire input carries a fresh
   ``configuration_update`` after the compaction item (Astra).
3. A new process resumes the saved session and recalls the codeword again.
4. ``fast-agent export --format atif`` produces a valid trajectory with an
   ``openai_compaction`` context-management boundary and the compaction call.

``--fixture-out`` copies the raw FAST_AGENT_LLM_TRACE capture of the compaction
request for promotion into ``tests/fixtures/llm_traces``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

CODEWORD = "PINEAPPLE-42"
FILLER = " ".join(f"line {i}: the quick brown fox jumps over the lazy dog." for i in range(1_500))
# fast-agent truncates the tool result to ~4k tokens; the tool request must cross this.
TRIGGER_TOKENS = 2_000
QUESTION = "What was the deployment codeword? Reply with it only."
FIRST_PROMPT = "Call read_notes, then reply only OK."
# xAI's compaction drops tool outputs, keeping what the conversation said: the reply
# restates the codeword so recall tests the checkpoint rather than the dropped output.
XAI_FIRST_PROMPT = "Call read_notes, then reply with only the deployment codeword it contains."


def read_notes() -> str:
    """Read the project notes file."""
    return f"Deployment codeword: {CODEWORD}.\n{FILLER}"


class Checks:
    def __init__(self, label: str) -> None:
        self.label = label
        self.failures = 0

    def warn(self, name: str, detail: object = "") -> None:
        print(f"  [WARN] {self.label}: {name}" + (f" — {detail}" if detail else ""))

    def check(self, name: str, ok: bool, detail: object = "") -> None:
        self.failures += not ok
        print(
            f"  [{'PASS' if ok else 'FAIL'}] {self.label}: {name}"
            + (f" — {detail}" if detail else "")
        )


ROUTE_PROVIDERS = {
    "codexresponses": "CODEX_RESPONSES",
    "copilot": "COPILOT",
    "responses": "RESPONSES",
    "xai": "XAI",
}


def write_config(workspace: Path, route: str, model: str) -> None:
    from fast_agent.llm.model_database import ModelDatabase
    from fast_agent.llm.provider_types import Provider

    provider = Provider[ROUTE_PROVIDERS[route]]
    window = ModelDatabase.get_context_window(model, provider=provider)
    assert window, f"no context window known for {model}"
    (workspace / "fastagent.config.yaml").write_text(
        "compaction:\n"
        f"  threshold: {TRIGGER_TOKENS / window:.4f}\n"
        "  keep_turns: 0\n"
        "logger:\n"
        "  show_chat: false\n"
        "  show_tools: false\n"
        "  progress_display: false\n",
        encoding="utf-8",
    )


def captured_requests(workspace: Path) -> list[tuple[Path, dict[str, Any]]]:
    captures = sorted((workspace / "stream-debug").glob("*_request.json"))
    return [(path, json.loads(path.read_text(encoding="utf-8"))) for path in captures]


def input_types(request: dict[str, Any]) -> list[str]:
    return [str(item.get("type") or item.get("role")) for item in request.get("input") or []]


def is_compaction_request(path: Path, request: dict[str, Any]) -> bool:
    """A ``compaction_trigger`` request, or a capture of xAI's ``/responses/compact``."""
    return "_compact_turn" in path.name or input_types(request)[-1:] == ["compaction_trigger"]


async def phase_run(model_spec: str, astra: bool, checks: Checks) -> None:
    from fast_agent import FastAgent
    from fast_agent.constants import OPENAI_COMPACTION_ITEMS
    from fast_agent.history.compaction import is_compaction_message
    from fast_agent.llm.provider.openai.responses_cache import read_cache_state
    from fast_agent.llm.reasoning_effort import parse_reasoning_setting

    fast = FastAgent(
        "codex-compaction-smoke",
        config_path="fastagent.config.yaml",
        parse_cli_args=False,
        quiet=True,
    )

    @fast.agent(
        name="agent", model=model_spec, instruction="You are terse.", function_tools=[read_notes]
    )
    async def _agent() -> None:
        pass

    async with fast.run() as app:
        agent = app.agent
        llm = agent.llm
        assert llm is not None
        turns = llm.usage_accumulator.turns
        await agent.send(XAI_FIRST_PROMPT if model_spec.startswith("xai.") else FIRST_PROMPT)
        prompts = [turn.prompt.total or 0 for turn in turns]
        tool_request_tokens = max(prompts)

        checkpoints = [m for m in agent.message_history if is_compaction_message(m)]
        checks.check(
            "turn compacted into exactly one checkpoint",
            len(checkpoints) == 1,
            f"{len(checkpoints)} checkpoint(s)",
        )
        if not checkpoints:
            return
        checkpoint = checkpoints[0]
        channels = checkpoint.channels or {}
        metadata = json.loads(channels["fast-agent-compaction"][0].text)  # ty: ignore[unresolved-attribute]
        checks.check(
            "checkpoint carries opaque compaction items",
            bool(channels.get(OPENAI_COMPACTION_ITEMS)),
        )
        checks.check(
            "checkpoint strategy is openai_compaction",
            metadata.get("strategy") == "openai_compaction",
            metadata.get("strategy"),
        )

        llm.set_reasoning_effort(parse_reasoning_setting("high"))
        start = len(turns)
        answer = await agent.send(QUESTION)
        follow_up_tokens = sum(turn.prompt.total or 0 for turn in turns[start:])
        checks.check("follow-up recalls the codeword", CODEWORD in answer, repr(answer))
        # Usage survives compaction, unlike the re-read tool call in history.
        reread = any(turn.tool_calls for turn in turns[start:])
        if reread:
            # Model-dependent: smaller models' checkpoints can drop the detail, so the
            # model re-reads the file (re-crossing the tiny trigger: more compactions).
            checks.warn(
                "follow-up re-read the file: the checkpoint did not keep the codeword",
                f"{tool_request_tokens} → {follow_up_tokens} prompt tokens",
            )
        else:
            checks.check(
                "follow-up input fell sharply",
                follow_up_tokens < tool_request_tokens * 0.25,
                f"{tool_request_tokens} → {follow_up_tokens} prompt tokens",
            )
        # Copilot sends effort per request; only managed routes record it per reply.
        if astra and "copilot." not in model_spec:
            state = read_cache_state(agent.message_history[-1])
            checks.check(
                "effort change recorded on the reply",
                state is not None and state.effort == "high",
                state.effort if state else None,
            )


async def phase_resume(model_spec: str, checks: Checks) -> None:
    from fast_agent import FastAgent
    from fast_agent.history.compaction import is_compaction_message
    from fast_agent.session import get_session_manager

    fast = FastAgent(
        "codex-compaction-smoke",
        config_path="fastagent.config.yaml",
        parse_cli_args=False,
        quiet=True,
    )

    @fast.agent(
        name="agent", model=model_spec, instruction="You are terse.", function_tools=[read_notes]
    )
    async def _agent() -> None:
        pass

    async with fast.run() as app:
        agent = app.agent
        result = await get_session_manager().resume_session_agents_async({"agent": agent})
        checks.check("saved session resumed", result is not None and "agent" in result.loaded)
        checks.check(
            "resumed history keeps the checkpoint",
            any(is_compaction_message(m) for m in agent.message_history),
        )
        answer = await agent.send("Repeat the deployment codeword only.")
        checks.check("resumed conversation recalls the codeword", CODEWORD in answer, repr(answer))


def check_wire(workspace: Path, astra: bool, checks: Checks) -> None:
    requests = captured_requests(workspace)
    trigger = [r for r in requests if is_compaction_request(*r)]
    checks.check(
        "compaction requests were sent",
        len(trigger) >= 1,
        f"{len(trigger)} request(s)",
    )
    if len(trigger) > 1:
        checks.warn("extra compactions after a re-read", f"{len(trigger)} compaction requests")
    follow = [
        r for r in requests if not is_compaction_request(*r) and "compaction" in input_types(r[1])
    ]
    checks.check("later requests replay the compaction item", bool(follow))
    if astra and follow:
        items = follow[0][1]["input"]
        types = input_types(follow[0][1])
        index = types.index("compaction")
        efforts = [
            item["reasoning"]["effort"]
            for item in items[index + 1 :]
            if item.get("type") == "configuration_update"
        ]
        checks.check(
            "fresh configuration_update with the new effort follows the compaction item",
            efforts == ["high"],
            f"{types} efforts={efforts}",
        )


def check_atif(workspace: Path, env: dict[str, str], checks: Checks) -> None:
    compactions = sum(is_compaction_request(*r) for r in captured_requests(workspace))
    from fast_agent.session.token_accounting_validation import validate_atif

    output = workspace / "trajectory.atif.json"
    completed = subprocess.run(
        ["fast-agent", "export", "latest", "--format", "atif", "--output", str(output)],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    checks.check(
        "persisted ATIF export succeeded",
        completed.returncode == 0,
        completed.stderr.strip()[-300:],
    )
    if completed.returncode != 0:
        return
    validate_atif(output)
    trajectory = json.loads(output.read_text(encoding="utf-8"))
    boundaries = [
        step
        for step in trajectory["steps"]
        if (step.get("extra") or {}).get("context_management", {}).get("type") == "compaction"
    ]
    checks.check(
        "ATIF has one boundary per compaction request",
        bool(boundaries) and len(boundaries) == compactions,
        f"{len(boundaries)} boundaries, {compactions} compaction requests",
    )
    if not boundaries:
        return
    strategies = [step["extra"]["context_management"].get("strategy") for step in boundaries]
    checks.check(
        "ATIF boundary strategies are openai_compaction",
        set(strategies) == {"openai_compaction"},
        strategies,
    )
    children = trajectory.get("subagent_trajectories") or []
    roles = [(child.get("extra") or {}).get("role") for child in children]
    checks.check("ATIF embeds the compaction call", "context_compaction_summary" in roles)
    checks.check(
        "ATIF notes explain the opaque checkpoint",
        "provider-native strategy" in (trajectory.get("notes") or ""),
    )
    extra = trajectory["final_metrics"].get("extra") or {}
    checks.check(
        "ATIF final metrics count the provider checkpoints",
        extra.get("provider_checkpoint_compactions") == len(boundaries),
        extra.get("provider_checkpoint_compactions"),
    )


def copy_fixture(workspace: Path, destination: Path) -> None:
    for path, request in captured_requests(workspace):
        if is_compaction_request(path, request):
            destination.mkdir(parents=True, exist_ok=True)
            base = path.name.removesuffix("_request.json")
            shutil.copy(path, destination / "request.json")
            shutil.copy(path.with_name(f"{base}_chunks.jsonl"), destination / "stream.jsonl")
            print(f"  raw compaction trace copied to {destination}")
            return


def run_transport(transport: str, args: argparse.Namespace) -> int:
    route, model = args.route, args.model
    workspace = Path(tempfile.mkdtemp(prefix=f"{route}-compaction-{transport}-"))
    print(f"\n== {route}.{model} over {transport} (workspace {workspace})")
    write_config(workspace, route, model)
    # Keep sessions inside the workspace, even when launched from a fast-agent shell.
    home = str(workspace / ".fast-agent")
    env = {
        **os.environ,
        "FAST_AGENT_LLM_TRACE": "1",
        "FAST_AGENT_HOME": home,
        "FAST_AGENT_RUNTIME_HOME": home,
    }
    env.pop("ENVIRONMENT_DIR", None)
    failures = 0
    for phase in ("run", "resume"):
        completed = subprocess.run(
            [
                sys.executable,
                __file__,
                "--phase",
                phase,
                "--transport",
                transport,
                "--route",
                route,
                "--model",
                model,
            ],
            cwd=workspace,
            env=env,
            check=False,
        )
        failures += completed.returncode != 0
    checks = Checks(f"{route}/{transport}")
    check_wire(workspace, model == "gpt-6-astra", checks)
    check_atif(workspace, env, checks)
    if args.fixture_out:
        copy_fixture(workspace, Path(args.fixture_out) / transport)
    return failures + checks.failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--transport", choices=("websocket", "sse", "both"), default="both")
    parser.add_argument("--route", choices=tuple(ROUTE_PROVIDERS), default="codexresponses")
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--fixture-out", help="Copy the raw compaction trace here")
    parser.add_argument("--phase", choices=("run", "resume"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.phase:
        checks = Checks(f"{args.route}/{args.transport}/{args.phase}")
        spec = f"{args.route}.{args.model}?transport={args.transport}"
        if args.phase == "run":
            asyncio.run(phase_run(spec, args.model == "gpt-6-astra", checks))
        else:
            asyncio.run(phase_resume(spec, checks))
        return 1 if checks.failures else 0

    transports = ("websocket", "sse") if args.transport == "both" else (args.transport,)
    failures = sum(run_transport(transport, args) for transport in transports)
    print("\nRESULT:", "PASS" if not failures else f"FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
