"""
Common constants and utilities shared between modules to avoid circular imports.
"""

import fnmatch
from collections.abc import Mapping, Sequence

# Constants
SEP = "__"


def create_namespaced_name(server_name: str, resource_name: str) -> str:
    """Create a namespaced resource name from server and resource names"""
    return f"{server_name}{SEP}{resource_name}"[:64]


def is_namespaced_name(name: str) -> bool:
    """Check if a name is already namespaced"""
    return SEP in name


def get_server_name(namespaced_name: str) -> str:
    """Extract the server name from a namespaced resource name"""
    return namespaced_name.split(SEP)[0] if SEP in namespaced_name else ""


def get_resource_name(namespaced_name: str) -> str:
    """Extract the resource name from a namespaced resource name"""
    return namespaced_name.split(SEP, 1)[1] if SEP in namespaced_name else namespaced_name


def matches_server_filter(
    filters: Mapping[str, Sequence[str]] | None, server_name: str, name: str
) -> bool:
    """Apply an agent filter: unlisted servers allow all; listed ones need a glob match."""
    patterns = (filters or {}).get(server_name)
    return patterns is None or any(fnmatch.fnmatch(name, pattern) for pattern in patterns)
