from __future__ import annotations

import difflib
import io
import os
import stat
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.error import YAMLError

from fast_agent.mcp.connect_targets import resolve_target_entry

_YAML = YAML()
_YAML.preserve_quotes = True


class MCPConfigMigrationError(ValueError):
    pass


def _mapping(value: object, path: str) -> CommentedMap:
    if not isinstance(value, CommentedMap):
        raise MCPConfigMigrationError(f"`{path}` must be a mapping")
    return value


def _move_comment(
    source: CommentedMap,
    source_key: object,
    destination: CommentedMap,
    destination_key: object,
) -> None:
    comment = source.ca.items.get(source_key)
    if comment is not None:
        destination.ca.items[destination_key] = comment


def _server_name(entry: object, index: int) -> tuple[str, CommentedMap]:
    source_path = f"mcp.targets[{index}]"
    if isinstance(entry, str):
        target = entry
        default_name = None
        overrides: dict[str, Any] = {}
        migrated = CommentedMap()
        migrated["target"] = entry
    elif isinstance(entry, CommentedMap):
        if "target" not in entry:
            raise MCPConfigMigrationError(f"`{source_path}.target` is required")
        target = entry["target"]
        if not isinstance(target, str):
            raise MCPConfigMigrationError(f"`{source_path}.target` must be a string")

        raw_name = entry.get("name")
        if raw_name is not None and not isinstance(raw_name, str):
            raise MCPConfigMigrationError(f"`{source_path}.name` must be a string")
        if isinstance(raw_name, str) and not raw_name.strip():
            raise MCPConfigMigrationError(f"`{source_path}.name` must be a non-empty string")
        default_name = raw_name
        overrides = {
            str(key): value for key, value in entry.items() if key not in {"target", "name"}
        }
        conflicting_fields = [
            field
            for field in ("transport", "url", "command", "args", "connector_id")
            if field in overrides
        ]
        if conflicting_fields:
            fields = ", ".join(conflicting_fields)
            raise MCPConfigMigrationError(
                f"`{source_path}.target` cannot be combined with source fields: {fields}"
            )
        migrated = entry
    else:
        raise MCPConfigMigrationError(f"`{source_path}` must be a string or mapping")

    try:
        resolved = resolve_target_entry(
            target=target,
            default_name=default_name,
            overrides=overrides,
            source_path=source_path,
        )
    except (TypeError, ValueError) as exc:
        raise MCPConfigMigrationError(str(exc)) from exc

    if "name" in migrated:
        del migrated["name"]
    return resolved.server_name, migrated


def _migrate_targets(mcp: CommentedMap) -> bool:
    if "targets" not in mcp:
        return False
    if "servers" in mcp:
        raise MCPConfigMigrationError("`mcp.targets` and `mcp.servers` cannot both be set")

    targets = mcp["targets"]
    if not isinstance(targets, CommentedSeq):
        raise MCPConfigMigrationError("`mcp.targets` must be a list")

    servers = CommentedMap()
    for index, entry in enumerate(targets):
        name_comment = entry.ca.items.get("name") if isinstance(entry, CommentedMap) else None
        name, migrated = _server_name(entry, index)
        if name in servers:
            raise MCPConfigMigrationError(
                f"`mcp.targets[{index}]` resolves to duplicate server name `{name}`"
            )
        servers[name] = migrated
        if name_comment is not None:
            servers.ca.items[name] = name_comment
        elif index in targets.ca.items:
            sequence_comment = targets.ca.items[index][0]
            if sequence_comment is not None:
                migrated.yaml_add_eol_comment(sequence_comment.value.strip(), key="target")

    index = list(mcp).index("targets")
    target_comment = mcp.ca.items.get("targets")
    del mcp["targets"]
    mcp.insert(index, "servers", servers)
    if target_comment is not None:
        mcp.ca.items["servers"] = target_comment
    return True


def _canonical_mcp(document: CommentedMap, *, required: bool) -> CommentedMap | None:
    if "mcp" not in document:
        if not required:
            return None
        mcp = CommentedMap()
        document["mcp"] = mcp
        return mcp

    raw_mcp = document["mcp"]
    if raw_mcp is None and required:
        mcp = CommentedMap()
        document["mcp"] = mcp
        return mcp
    if raw_mcp is None:
        return None
    return _mapping(raw_mcp, "mcp")


def _move_legacy_setting(
    document: CommentedMap,
    mcp: CommentedMap,
    *,
    legacy_key: str,
    section_name: str,
    canonical_key: str,
) -> bool:
    if legacy_key not in document:
        return False

    if section_name in mcp:
        section = _mapping(mcp[section_name], f"mcp.{section_name}")
    else:
        section = CommentedMap()
        mcp[section_name] = section

    canonical_path = f"mcp.{section_name}.{canonical_key}"
    if canonical_key in section:
        raise MCPConfigMigrationError(f"`{legacy_key}` and `{canonical_path}` cannot both be set")

    section[canonical_key] = document[legacy_key]
    _move_comment(document, legacy_key, section, canonical_key)
    del document[legacy_key]
    return True


def migrate_mcp_document(document: object) -> tuple[CommentedMap, bool]:
    root = _mapping(document, "configuration")
    legacy_settings = "auto_sampling" in root or "mcp_timeline" in root
    mcp = _canonical_mcp(root, required=legacy_settings)

    changed = False
    if mcp is not None:
        changed |= _migrate_targets(mcp)
        changed |= _move_legacy_setting(
            root,
            mcp,
            legacy_key="auto_sampling",
            section_name="client",
            canonical_key="auto_sampling",
        )
        changed |= _move_legacy_setting(
            root,
            mcp,
            legacy_key="mcp_timeline",
            section_name="diagnostics",
            canonical_key="timeline",
        )

    return root, changed


def load_and_migrate_mcp(path: Path) -> tuple[bytes, str, bool]:
    try:
        original = path.read_bytes()
        text = original.decode("utf-8")
        document = _YAML.load(text)
    except UnicodeDecodeError as exc:
        raise MCPConfigMigrationError("configuration must be UTF-8") from exc
    except YAMLError as exc:
        problem = getattr(exc, "problem", None)
        raise MCPConfigMigrationError(f"invalid YAML: {problem or exc}") from exc

    if document is None:
        document = CommentedMap()
    migrated, changed = migrate_mcp_document(document)
    if not changed:
        return original, text, False

    stream = io.StringIO()
    _YAML.dump(migrated, stream)
    return original, stream.getvalue(), True


def unified_mcp_diff(path: Path, original: bytes, migrated: str) -> str:
    source = original.decode("utf-8")
    return "".join(
        difflib.unified_diff(
            source.splitlines(keepends=True),
            migrated.splitlines(keepends=True),
            fromfile=str(path),
            tofile=str(path),
        )
    )


def _atomic_replace(path: Path, contents: bytes, mode: int) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            with suppress(OSError):
                temporary.unlink()


def write_mcp_migration(path: Path, original: bytes, migrated: str) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    _atomic_replace(path.with_name(f"{path.name}.bak"), original, mode)
    _atomic_replace(path, migrated.encode("utf-8"), mode)


__all__ = [
    "MCPConfigMigrationError",
    "load_and_migrate_mcp",
    "migrate_mcp_document",
    "unified_mcp_diff",
    "write_mcp_migration",
]
