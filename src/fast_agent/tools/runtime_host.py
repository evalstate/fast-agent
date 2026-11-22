"""What an agent supplies to the tool runtimes it owns.

Runtimes are bound to their agent once, instead of receiving a setter per callback. Guidance
the tool engine reports (how the model reads files now; how it reaches tools under tool search
later) is added here as a member rather than as another setter at every binding site.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from fast_agent.mcp.tool_execution_handler import ToolExecutionHandler
    from fast_agent.tools.file_read_access import FileReadAccess
    from fast_agent.types import RequestParams


class FileReadAccessProvider(Protocol):
    @property
    def file_read_access(self) -> FileReadAccess:
        """How the model reads files; must not depend on the runtimes' tool lists."""
        ...


class RuntimeHost(FileReadAccessProvider, Protocol):
    def tool_handler_for(
        self, request_params: RequestParams | None = None
    ) -> ToolExecutionHandler | None: ...
