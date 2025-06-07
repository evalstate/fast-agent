"""
Common constants and utilities shared between modules to avoid circular imports.
"""

# Constants
SEP = "-"


def create_namespaced_name(server_name: str, resource_name: str) -> str:
    """Create a namespaced resource name from server and resource names"""
    return f"{server_name}{SEP}{resource_name}"[:64]


def is_namespaced_name(name: str) -> bool:
    """Check if a name is already namespaced"""
    return SEP in name
