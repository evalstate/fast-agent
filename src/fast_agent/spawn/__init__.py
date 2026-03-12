"""Dynamic Agent Spawn — subprocess-based agent lifecycle management.

Core capabilities:

- **Single agent spawn**: isolated subprocess with full FastAgent instance
- **Background spawn**: fire-and-forget with status polling
- **Team spawn**: template-driven multi-agent orchestration with DAG deps
- **Review loops**: iterative review with escalation policies
- **Meetings**: multi-agent concurrent discussion with turn management
- **Registry**: file-based tracking of all spawned agent lifecycles
- **Signals**: cross-process completion notification
- **Message bus**: file-based inter-agent communication
"""

from __future__ import annotations
