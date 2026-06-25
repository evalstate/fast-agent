#!/usr/bin/env python3
"""Generate Hugging Face API workflow agent cards from one canonical body."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BODY_PATH = ROOT / "scripts" / "hf_api_agent_body.md"

CARDS = {
    "hf-api-agent-small.md": """---
type: agent
name: hf-user-small
function_tools:
  - hf_api_tool.py:hf_api_request
model: hf.openai/gpt-oss-20b:groq
default: true
description: Use this tool to find out information about Users, Organizations and Pull Requests
---
""",
    "hf_hub_community.md": """---
function_tools:
  - hf_api_tool.py:hf_api_request
model: gpt-oss
description: "Query Hugging Face community features: user/org profiles, followers, repo discussions, pull requests, comments, access requests, and collections. Use for people lookups and repo collaboration—not for model/dataset search."
---
""",
}


def modernize_signature_text(text: str) -> str:
    """Normalize legacy typing spellings in generated snippets."""
    return text.replace("Optional[str]", "str | None")


def main() -> None:
    body = modernize_signature_text(BODY_PATH.read_text())
    for filename, frontmatter in CARDS.items():
        (ROOT / filename).write_text(frontmatter + body)


if __name__ == "__main__":
    main()
