"""
XML formatting utilities for consistent prompt engineering across components.
"""

def format_xml_tag(
    tag_name: str,
    content: str | None = None,
    attributes: dict[str, str] | None = None,
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
    content: str | None = None,
    attributes: dict[str, str] | None = None,
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

