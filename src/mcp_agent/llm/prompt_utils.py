"""
XML formatting utilities for consistent prompt engineering across components.
"""

from typing import Dict, List, Optional, Union


def format_xml_tag(
    tag_name: str,
    content: Optional[str] = None,
    attributes: Optional[Dict[str, str]] = None,
) -> str:
    """
    Format an XML tag with optional content and attributes.
    Uses self-closing tag when content is None or empty.

    Args:
        tag_name: Name of the XML tag
        content: Content to include inside the tag (None for self-closing)
        attributes: Dictionary of attribute name-value pairs

    Returns:
        Formatted XML tag as string
    """
    # Format attributes if provided
    attrs_str = ""
    if attributes:
        attrs_str = " " + " ".join(f'{k}="{v}"' for k, v in attributes.items())

    # Use self-closing tag if no content
    if content is None or content == "":
        return f"<{tag_name}{attrs_str} />"

    # Full tag with content
    return f"<{tag_name}{attrs_str}>{content}</{tag_name}>"


def format_fastagent_tag(
    tag_type: str,
    content: Optional[str] = None,
    attributes: Optional[Dict[str, str]] = None,
) -> str:
    """
    Format a fastagent-namespaced XML tag with consistent formatting.

    Args:
        tag_type: Type of fastagent tag (without namespace prefix)
        content: Content to include inside the tag
        attributes: Dictionary of attribute name-value pairs

    Returns:
        Formatted fastagent XML tag as string
    """
    return format_xml_tag(f"fastagent:{tag_type}", content, attributes)


def format_server_info(
    server_name: str,
    description: Optional[str] = None,
    tools: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Format server information consistently across router and orchestrator modules.

    Args:
        server_name: Name of the server
        description: Optional server description
        tools: Optional list of tool dictionaries with 'name' and 'description' keys

    Returns:
        Formatted server XML as string
    """
    # Use self-closing tag if no description or tools
    if not description and not tools:
        return format_fastagent_tag("server", None, {"name": server_name})

    # Start building components
    components = []

    # Add description if present
    if description:
        desc_tag = format_fastagent_tag("description", description)
        components.append(desc_tag)

    # Add tools section if tools exist
    if tools and len(tools) > 0:
        tool_tags = []
        for tool in tools:
            tool_name = tool.get("name", "")
            tool_desc = tool.get("description", "")
            tool_tag = format_fastagent_tag("tool", tool_desc, {"name": tool_name})
            tool_tags.append(tool_tag)

        tools_content = "\n".join(tool_tags)
        tools_tag = format_fastagent_tag("tools", f"\n{tools_content}\n")
        components.append(tools_tag)

    # Combine all components
    server_content = "\n".join(components)
    return format_fastagent_tag("server", f"\n{server_content}\n", {"name": server_name})


def format_agent_info(
    agent_name: str,
    description: Optional[str] = None,
    servers: Optional[List[Dict[str, Union[str, List[Dict[str, str]]]]]] = None,
) -> str:
    """
    Format agent information consistently across router and orchestrator modules.

    Args:
        agent_name: Name of the agent
        description: Optional agent description/instruction
        servers: Optional list of server dictionaries with 'name', 'description', and 'tools' keys

    Returns:
        Formatted agent XML as string
    """
    # Start building components
    components = []

    # Add description if present
    if description:
        desc_tag = format_fastagent_tag("description", description)
        components.append(desc_tag)

    # If no description or servers, use self-closing tag
    if not description and not servers:
        return format_fastagent_tag("agent", None, {"name": agent_name})

    # If has servers, format them
    if servers and len(servers) > 0:
        server_tags = []
        for server in servers:
            server_name = server.get("name", "")
            server_desc = server.get("description", "")
            server_tools = server.get("tools", [])
            server_tag = format_server_info(server_name, server_desc, server_tools)
            server_tags.append(server_tag)

        # Only add servers section if we have servers
        if server_tags:
            servers_content = "\n".join(server_tags)
            servers_tag = format_fastagent_tag("servers", f"\n{servers_content}\n")
            components.append(servers_tag)

    # Combine all components
    agent_content = "\n".join(components)
    return format_fastagent_tag("agent", f"\n{agent_content}\n", {"name": agent_name})
