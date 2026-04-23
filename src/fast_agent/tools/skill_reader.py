"""
SkillReader - Read skill files for non-ACP contexts.

This provides a dedicated 'read_skill' tool for reading SKILL.md files and
associated resources when not running in an ACP context (where read_text_file
is provided by ACPFilesystemRuntime).

Also handles Skills-over-MCP URIs (any `<scheme>://...` that descends from
a discovered manifest root — `skill://` is the SEP's default but not
required) by dispatching through the MCP aggregator, so filesystem-backed
and MCP-backed skills share one tool.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.types import BlobResourceContents, CallToolResult, TextContent, TextResourceContents, Tool

from fast_agent.mcp.skill_uri import strip_skill_md

if TYPE_CHECKING:
    from fast_agent.mcp.mcp_aggregator import MCPAggregator
    from fast_agent.skills.registry import SkillManifest


class SkillReader:
    """Provides the read_skill tool for reading skill files in non-ACP contexts."""

    def __init__(
        self,
        skill_manifests: list[SkillManifest],
        logger,
        *,
        aggregator: "MCPAggregator | None" = None,
    ) -> None:
        """
        Initialize the skill reader.

        Args:
            skill_manifests: List of available skill manifests (for path validation)
            logger: Logger instance for debugging
            aggregator: MCP aggregator for reading Skills-over-MCP resources.
                Required when any manifest is URI-backed; optional otherwise.
        """
        self._skill_manifests = skill_manifests
        self._logger = logger
        self._aggregator = aggregator

        # Build set of allowed filesystem skill directories (for path reads)
        self._allowed_directories: set[Path] = set()
        # Build set of allowed URI roots (for URI reads). A read is allowed
        # if the requested URI begins with one of these roots — same trust
        # boundary as the filesystem allowed-directories check. Any scheme
        # is accepted per the SEP (`skill://`, `github://`, `repo://`, ...).
        self._allowed_uri_roots: set[str] = set()
        for manifest in skill_manifests:
            if manifest.path:
                self._allowed_directories.add(manifest.path.parent.resolve())
            if manifest.uri:
                self._allowed_uri_roots.add(strip_skill_md(manifest.uri))

        self._tool = Tool(
            name="read_skill",
            description=(
                "Read a skill's SKILL.md file or associated resources. "
                "Use this to load skill instructions before using the skill."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Absolute filesystem path or resource URI of the file to "
                            "read. Pass whatever appears in <location> verbatim — most "
                            "often a `skill://...` URI for MCP-served skills, though "
                            "other schemes (`github://`, `repo://`) are valid per the "
                            "SEP. Filesystem skills use absolute paths."
                        ),
                    }
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        )

    @property
    def tool(self) -> Tool:
        """Get the read_skill tool definition."""
        return self._tool

    @property
    def enabled(self) -> bool:
        """Whether the skill reader is enabled (has skills available)."""
        return len(self._skill_manifests) > 0

    def _is_path_allowed(self, path: Path) -> bool:
        """Check if the path is within an allowed skill directory."""
        resolved = path.resolve()
        for allowed_dir in self._allowed_directories:
            try:
                resolved.relative_to(allowed_dir)
                return True
            except ValueError:
                continue
        return False

    def _is_uri_allowed(self, uri: str) -> bool:
        """Check the URI is under a known skill root (trust boundary).

        Rejects URIs containing `..` or `.` path segments (raw or
        percent-encoded) so `skill://good/../evil/SKILL.md` and
        `skill://good/%2E%2E/evil/SKILL.md` cannot slip past the
        prefix check. Also rejects query (`?`) and fragment (`#`)
        suffixes — neither is meaningful for skill resource reads and
        leaving them in would let a caller sidestep the exact-match
        path by appending junk. The filesystem guard relies on
        `Path.resolve()` for the same normalization; URIs get these
        explicit rejects instead of a full URL normalize to keep the
        trust boundary independent of server URI semantics.

        Case handling follows RFC 3986: scheme and traversal-marker
        checks operate on a lowercased copy (scheme is case-insensitive,
        and we want `%2E` / `%2e` both caught); the root-prefix check
        compares raw URIs because the path component is case-sensitive.
        A server publishing `skill://Acme/...` will not match a model
        call for `skill://acme/...` — publish consistently.
        """
        if "?" in uri or "#" in uri:
            return False
        lowered = uri.lower()
        # Defense in depth: even if a manifest somehow registered a `file://`
        # root, refuse to honor it here. The loader already rejects `file://`
        # entries; this is the fallback trust-boundary check.
        if lowered.startswith("file://"):
            return False
        # Check each path segment (after the scheme) for raw or
        # percent-encoded traversal markers.
        scheme_sep = lowered.find("://")
        tail = lowered[scheme_sep + 3 :] if scheme_sep != -1 else lowered
        for segment in tail.split("/"):
            decoded = segment.replace("%2e", ".")
            if decoded in ("..", "."):
                return False
        for root in self._allowed_uri_roots:
            if uri == root or uri.startswith(f"{root}/"):
                return True
        return False

    def _find_server_for_uri(self, uri: str) -> str | None:
        """Return the MCP server that serves the skill covering this URI."""
        best_len = -1
        best_server: str | None = None
        for manifest in self._skill_manifests:
            if not manifest.uri or not manifest.server_name:
                continue
            root = strip_skill_md(manifest.uri)
            if uri == root or uri.startswith(f"{root}/") or uri == manifest.uri:
                # Prefer the most specific (longest) match when roots overlap.
                if len(root) > best_len:
                    best_len = len(root)
                    best_server = manifest.server_name
        return best_server

    async def execute(self, arguments: dict[str, Any] | None = None) -> CallToolResult:
        """Read a skill file (filesystem path or any resource URI)."""
        path_str = (arguments or {}).get("path") if arguments else None
        if not isinstance(path_str, str) or not path_str.strip():
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text="The read_skill tool requires a 'path' string argument.",
                    )
                ],
            )

        target = path_str.strip()
        if _looks_like_uri(target):
            return await self._read_mcp_uri(target)
        return await self._read_filesystem(target)

    async def _read_mcp_uri(self, uri: str) -> CallToolResult:
        if self._aggregator is None:
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=(
                            "No MCP aggregator is configured to resolve URI-based "
                            "skill resources for this agent."
                        ),
                    )
                ],
            )

        if not self._is_uri_allowed(uri):
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Access denied: {uri} is not within an allowed skill root.",
                    )
                ],
            )

        server_name = self._find_server_for_uri(uri)
        if server_name is None:
            # Defense in depth: the manifest invariant should guarantee every
            # URI has a publishing server, but if that invariant is ever
            # violated the aggregator would iterate every connected server
            # and return the first one that happens to serve the URI —
            # crossing the per-server trust boundary silently.
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=(
                            f"Access denied: {uri} is not mapped to a known "
                            "MCP server."
                        ),
                    )
                ],
            )
        try:
            result = await self._aggregator.get_resource(uri, server_name=server_name)
        except Exception as exc:
            self._logger.error(
                "Failed to read MCP skill resource",
                data={"uri": uri, "server": server_name, "error": str(exc)},
            )
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text=f"Error reading resource: {exc}")],
            )

        text_parts: list[str] = []
        binary_mimes: list[str] = []
        for item in result.contents:
            if isinstance(item, TextResourceContents):
                text_parts.append(item.text)
            elif isinstance(item, BlobResourceContents):
                binary_mimes.append(item.mimeType or "application/octet-stream")

        if not text_parts:
            if binary_mimes:
                # read_skill exists to load skill *text* (SKILL.md, references,
                # scripts). A blob-only resource has no text the model can act
                # on — return an error rather than synthesizing a fake text
                # placeholder the model would treat as content.
                mimes = ", ".join(sorted(set(binary_mimes)))
                return CallToolResult(
                    isError=True,
                    content=[
                        TextContent(
                            type="text",
                            text=(
                                f"Resource {uri} is binary ({mimes}); "
                                f"read_skill only returns text content."
                            ),
                        )
                    ],
                )
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Resource returned no content: {uri}",
                    )
                ],
            )

        self._logger.debug(
            "Read MCP skill resource",
            data={"uri": uri, "chars": sum(len(p) for p in text_parts)},
        )
        return CallToolResult(
            isError=False,
            content=[TextContent(type="text", text="\n".join(text_parts))],
        )

    async def _read_filesystem(self, path_str: str) -> CallToolResult:
        path = Path(path_str)

        # Security: ensure path is absolute
        if not path.is_absolute():
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text="Path must be absolute. Use the path from <location> in available_skills.",
                    )
                ],
            )

        # Security: ensure path is within an allowed skill directory
        if not self._is_path_allowed(path):
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Access denied: {path} is not within an allowed skill directory.",
                    )
                ],
            )

        # Check file exists
        if not path.exists():
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"File not found: {path}",
                    )
                ],
            )

        if not path.is_file():
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Path is not a file: {path}",
                    )
                ],
            )

        try:
            content = path.read_text(encoding="utf-8")
            self._logger.debug(
                "Read skill file",
                data={"path": str(path), "bytes": len(content)},
            )

            return CallToolResult(
                isError=False,
                content=[
                    TextContent(
                        type="text",
                        text=content,
                    )
                ],
            )
        except Exception as exc:
            self._logger.error(
                "Failed to read skill file",
                data={"path": str(path), "error": str(exc)},
            )
            return CallToolResult(
                isError=True,
                content=[
                    TextContent(
                        type="text",
                        text=f"Error reading file: {exc}",
                    )
                ],
            )


def _looks_like_uri(value: str) -> bool:
    """Detect a URI by `<scheme>://` shape.

    Per the SEP, `skill://` is a SHOULD: servers MAY publish skills under
    any scheme (`github://`, `repo://`, etc.) so long as they're listed in
    `skill://index.json`. The reader routes any URI through the aggregator
    and lets `_is_uri_allowed` enforce that it descends from a discovered
    skill root.
    """
    sep = value.find("://")
    if sep <= 0:
        return False
    # The scheme part must be a plain identifier (alpha + a few specials per
    # RFC 3986). Reject Windows drive paths like `C://...` is not an issue
    # since drive letters are single chars (`C:\\` not `C://`), but an
    # over-eager `://` substring on something like `https//x` shouldn't match
    # either — `find` returns -1 for that. This guard is defensive: only
    # treat as a URI if the scheme contains alphanumerics/+/-/.
    scheme = value[:sep]
    if not scheme or not all(c.isalnum() or c in "+-." for c in scheme):
        return False
    return True
