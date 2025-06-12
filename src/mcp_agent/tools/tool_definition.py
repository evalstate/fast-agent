from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ToolDefinition:
    """
    Represents a definition of a tool available to the agent.
    """

    name: str
    description: Optional[str] = None
    inputSchema: Dict[str, Any] = field(default_factory=dict)
    # Add other relevant fields if necessary based on how tools are defined in fast-agent
