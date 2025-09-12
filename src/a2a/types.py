"""
Lightweight compatibility types for A2A when the external package is unavailable.

These dataclasses provide just enough structure for integration and tests.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class AgentCapabilities:
    streaming: bool = False
    push_notifications: bool = False
    state_transition_history: bool = False


@dataclass
class AgentSkill:
    id: str
    name: str
    description: str = ""
    tags: Optional[List[str]] = None
    examples: Optional[List[str]] = None
    input_modes: Optional[List[str]] = None
    output_modes: Optional[List[str]] = None


@dataclass
class AgentCard:
    skills: List[AgentSkill]
    name: str
    description: str
    url: str
    version: str
    capabilities: Optional[AgentCapabilities] = None
    default_input_modes: Optional[List[str]] = None
    default_output_modes: Optional[List[str]] = None
    provider: Optional[str] = None
    documentation_url: Optional[str] = None
