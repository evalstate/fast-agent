"""Dynamic Agent Spawn — subprocess-based agent lifecycle management.

Core capabilities:

- **Single agent spawn**: isolated subprocess with full FastAgent instance
- **Background spawn**: fire-and-forget with status polling
- **Team spawn**: template-driven multi-agent orchestration
- **Meetings**: multi-agent concurrent discussion with turn management
- **Registry**: file-based tracking of all spawned agent lifecycles
- **Message bus**: file-based inter-agent communication with CC support
"""

from __future__ import annotations
