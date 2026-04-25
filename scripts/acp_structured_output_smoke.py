from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from acp.exceptions import RequestError
from acp.helpers import text_block
from acp.schema import ClientCapabilities, FileSystemCapabilities, Implementation
from acp.stdio import spawn_agent_process
from jsonschema.validators import validator_for

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
    },
    "required": ["answer"],
    "additionalProperties": False,
}


class SmokeClient:
    def __init__(self) -> None:
        self.notifications: list[dict[str, Any]] = []

    async def session_update(
        self,
        session_id: str,
        update: Any,
        **kwargs: Any,
    ) -> None:
        self.notifications.append({"session_id": session_id, "update": update})


def _message_chunks(client: SmokeClient, session_id: str, *, start_index: int = 0) -> list[str]:
    chunks: list[str] = []
    for notification in client.notifications[start_index:]:
        if notification["session_id"] != session_id:
            continue
        update = notification["update"]
        update_type = getattr(update, "sessionUpdate", None)
        if update_type != "agent_message_chunk":
            continue
        content = getattr(update, "content", None)
        text = getattr(content, "text", None)
        if isinstance(text, str):
            chunks.append(text)
    return chunks


def _validate_json_payload(text: str) -> None:
    payload = json.loads(text)
    validator_class = validator_for(SCHEMA)
    validator_class.check_schema(SCHEMA)
    validator_class(SCHEMA).validate(payload)


async def _wait_for_agent_message_chunk(
    client: SmokeClient,
    session_id: str,
    *,
    start_index: int = 0,
    timeout: float = 2.0,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        for notification in client.notifications[start_index:]:
            if notification["session_id"] != session_id:
                continue
            update_type = getattr(notification["update"], "sessionUpdate", None)
            if update_type == "agent_message_chunk":
                return
        await asyncio.sleep(0.05)
    raise AssertionError(f"Expected agent_message_chunk for session {session_id}")


async def run_smoke(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "-m",
        "fast_agent.cli",
        "serve",
        "--transport",
        "acp",
        "--model",
        args.model,
        "--name",
        "acp-structured-output-smoke",
        "--no-permissions",
    ]
    if args.config_path:
        cmd.extend(["--config-path", args.config_path])
    if args.env_dir:
        cmd.extend(["--env", args.env_dir])

    client = SmokeClient()
    async with spawn_agent_process(
        lambda _: client,
        *cmd,
        env=os.environ.copy(),
    ) as (connection, _process):
        try:
            init_response = await connection.initialize(
                protocol_version=1,
                client_capabilities=ClientCapabilities(
                    fs=FileSystemCapabilities(read_text_file=False, write_text_file=False),
                    terminal=False,
                ),
                client_info=Implementation(name="acp-json-smoke", version="0.0.1"),
            )
            capability_meta = init_response.agent_capabilities.field_meta
            if not capability_meta:
                raise AssertionError("initialize response did not include agentCapabilities._meta")
            if capability_meta.get("co.huggingface", {}).get("structuredOutput") is not True:
                raise AssertionError("structuredOutput capability was not advertised")

            session = await connection.new_session(mcp_servers=[], cwd=str(Path.cwd()))
            session_id = session.session_id
            if not session_id:
                raise AssertionError("session/new did not return a session id")

            normal_start = len(client.notifications)
            normal_response = await connection.prompt(
                session_id=session_id,
                prompt=[text_block(args.normal_prompt)],
            )
            await _wait_for_agent_message_chunk(client, session_id, start_index=normal_start)
            normal_text = "".join(_message_chunks(client, session_id, start_index=normal_start))
            print(f"normal stopReason: {normal_response.stop_reason}")
            print(f"normal text: {normal_text}")

            structured_meta: dict[str, Any] = {
                "co.huggingface": {
                    "structuredOutput": {
                        "schema": SCHEMA,
                        "mode": "bestEffort",
                    }
                }
            }
            structured_start = len(client.notifications)
            structured_response = await connection.prompt(
                session_id=session_id,
                prompt=[text_block(args.structured_prompt)],
                **structured_meta,
            )
            await _wait_for_agent_message_chunk(
                client,
                session_id,
                start_index=structured_start,
            )
            structured_text = "".join(
                _message_chunks(client, session_id, start_index=structured_start)
            )
            print(f"structured stopReason: {structured_response.stop_reason}")
            print(f"structured text: {structured_text}")

            _validate_json_payload(structured_text)
            print("structured JSON validated against supplied schema")
        except RequestError as exc:
            print(f"ACP request failed: {exc}", file=sys.stderr)
            if exc.data is not None:
                print(json.dumps(exc.data, indent=2), file=sys.stderr)
            if exc.code == -32000:
                print(
                    "Authentication is required for this model. Check the provider key "
                    "env vars/secrets advertised above, or use --model passthrough for "
                    "a credential-free transport smoke.",
                    file=sys.stderr,
                )
            return 1

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal ACP e2e smoke for co.huggingface.structuredOutput."
    )
    parser.add_argument("--model", default="passthrough")
    parser.add_argument("--config-path", help="Optional fastagent.config.yaml path")
    parser.add_argument("--env-dir", help="Optional fast-agent environment directory")
    parser.add_argument("--normal-prompt", default="hello from ACP smoke")
    parser.add_argument("--structured-prompt", default='{"answer":"ok"}')
    return parser.parse_args()


def main() -> int:
    return asyncio.run(run_smoke(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
