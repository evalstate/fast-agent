"""How the model reads local text files, as reported by the agent's tool engine.

Model-facing guidance (the skills prompt, tool descriptions, retained-output notices) names a
file reader. Tool engines change which readers the model can reach: with programmatic tool
calling, sandbox code reads files with ``open()`` and ``read_text_file`` is not offered. The
agent reports its access here so that guidance only names readers the model can use.
"""

from __future__ import annotations

from dataclasses import dataclass

from fast_agent.tools.filesystem_tool_definitions import READ_TEXT_FILE_TOOL_NAME
from fast_agent.tools.skill_reader import READ_SKILL_TOOL_NAME


@dataclass(frozen=True, slots=True)
class ToolFileReads:
    """The model reads files by calling ``tool``."""

    tool: str = READ_TEXT_FILE_TOOL_NAME

    @property
    def reader(self) -> str:
        return self.tool


@dataclass(frozen=True, slots=True)
class SandboxFileReads:
    """Code run by ``code_tool`` reads files with Python's ``open()``."""

    code_tool: str

    @property
    def reader(self) -> str:
        return "`open()`"


type FileReadAccess = ToolFileReads | SandboxFileReads

TOOL_FILE_READS = ToolFileReads()
SKILL_READER_FILE_READS = ToolFileReads(tool=READ_SKILL_TOOL_NAME)
"""Skills only: the ``read_skill`` fallback when no file reader is exposed."""
