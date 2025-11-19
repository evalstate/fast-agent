"""
Constants for the prompt system.

This module defines constants used throughout the prompt system, including
delimiters for parsing prompt files and serializing prompt messages.
"""

# Standard delimiters used for prompt template parsing and serialization
USER_DELIMITER = "---USER"
ASSISTANT_DELIMITER = "---ASSISTANT"
RESOURCE_DELIMITER = "---RESOURCE"

# Default delimiter mapping used by PromptTemplate and PromptTemplateLoader
DEFAULT_DELIMITER_MAP = {
    USER_DELIMITER: "user",
    ASSISTANT_DELIMITER: "assistant",
    RESOURCE_DELIMITER: "resource",
}
