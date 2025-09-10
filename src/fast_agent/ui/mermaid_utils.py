"""Utilities for detecting and processing Mermaid diagrams in text content."""

import base64
import re
import zlib
from dataclasses import dataclass
from typing import List, Optional

# Mermaid chart viewer URL prefix
MERMAID_VIEWER_URL = "https://www.mermaidchart.com/play#"
# mermaid.live#pako= also works but the playground has better ux


@dataclass
class MermaidDiagram:
    """Represents a detected Mermaid diagram."""

    content: str
    title: Optional[str] = None
    start_pos: int = 0
    end_pos: int = 0


def extract_mermaid_diagrams(text: str) -> List[MermaidDiagram]:
    """
    Extract all Mermaid diagram blocks from text content.

    Handles both simple mermaid blocks and blocks with titles:
    - ```mermaid
    - ```mermaid title={Some Title}

    Also extracts titles from within the diagram content.

    Args:
        text: The text content to search for Mermaid diagrams

    Returns:
        List of MermaidDiagram objects found in the text
    """
    diagrams = []

    # Pattern to match mermaid code blocks with optional title
    # Matches: ```mermaid or ```mermaid title={...}
    pattern = r"```mermaid(?:\s+title=\{([^}]+)\})?\s*\n(.*?)```"

    for match in re.finditer(pattern, text, re.DOTALL):
        title = match.group(1)  # May be None if no title
        content = match.group(2).strip()

        if content:  # Only add if there's actual diagram content
            # If no title from code fence, look for title in the content
            if not title:
                # Look for various title patterns in mermaid diagrams
                # pie title, graph title, etc.
                title_patterns = [
                    r"^\s*title\s+(.+?)(?:\n|$)",  # Generic title
                    r"^\s*pie\s+title\s+(.+?)(?:\n|$)",  # Pie chart title
                    r"^\s*gantt\s+title\s+(.+?)(?:\n|$)",  # Gantt chart title
                ]

                for title_pattern in title_patterns:
                    title_match = re.search(title_pattern, content, re.MULTILINE)
                    if title_match:
                        title = title_match.group(1).strip()
                        break

            diagrams.append(
                MermaidDiagram(
                    content=content, title=title, start_pos=match.start(), end_pos=match.end()
                )
            )

    return diagrams


def create_mermaid_live_link(diagram_content: str) -> str:
    """
    Create a Mermaid Live Editor link from diagram content.

    The link uses pako compression (zlib) and base64 encoding.

    Args:
        diagram_content: The Mermaid diagram source code

    Returns:
        Complete URL to Mermaid Live Editor
    """
    # Create the JSON structure expected by Mermaid Live
    # Escape newlines and quotes in the diagram content
    escaped_content = diagram_content.replace('"', '\\"').replace("\n", "\\n")
    json_str = f'{{"code":"{escaped_content}","mermaid":{{"theme":"default"}},"updateEditor":false,"autoSync":true,"updateDiagram":false}}'

    # Compress using zlib (pako compatible)
    compressed = zlib.compress(json_str.encode("utf-8"))

    # Base64 encode
    encoded = base64.urlsafe_b64encode(compressed).decode("utf-8")

    # Remove padding characters as Mermaid Live doesn't use them
    encoded = encoded.rstrip("=")

    return f"{MERMAID_VIEWER_URL}pako:{encoded}"


def format_mermaid_links(diagrams: List[MermaidDiagram]) -> List[str]:
    """
    Format Mermaid diagrams as markdown links.

    Args:
        diagrams: List of MermaidDiagram objects

    Returns:
        List of formatted markdown strings
    """
    links = []

    for i, diagram in enumerate(diagrams, 1):
        link = create_mermaid_live_link(diagram.content)

        if diagram.title:
            # Use the title from the diagram with number
            markdown = f"Diagram {i} - {diagram.title}: [Open Diagram]({link})"
        else:
            # Use generic numbering
            markdown = f"Diagram {i}: [Open Diagram]({link})"

        links.append(markdown)

    return links


def detect_diagram_type(content: str) -> str:
    """
    Detect the type of mermaid diagram from content.

    Args:
        content: The mermaid diagram source code

    Returns:
        Human-readable diagram type name
    """
    content_lower = content.strip().lower()

    # Check for common diagram types
    if content_lower.startswith(("graph ", "flowchart ")):
        return "Flowchart"
    elif content_lower.startswith("sequencediagram"):
        return "Sequence"
    elif content_lower.startswith("pie"):
        return "Pie Chart"
    elif content_lower.startswith("gantt"):
        return "Gantt Chart"
    elif content_lower.startswith("classdiagram"):
        return "Class Diagram"
    elif content_lower.startswith("statediagram"):
        return "State Diagram"
    elif content_lower.startswith("erdiagram"):
        return "ER Diagram"
    elif content_lower.startswith("journey"):
        return "User Journey"
    elif content_lower.startswith("gitgraph"):
        return "Git Graph"
    elif content_lower.startswith("c4context"):
        return "C4 Context"
    elif content_lower.startswith("mindmap"):
        return "Mind Map"
    elif content_lower.startswith("timeline"):
        return "Timeline"
    else:
        return "Diagram"
