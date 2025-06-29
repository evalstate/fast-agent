"""
FastMCP Prompt Server V2

A server that loads prompts from text files with simple delimiters and serves them via MCP.
Uses the prompt_template module for clean, testable handling of prompt templates.
"""

import argparse
import asyncio
import base64
import logging
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts.base import (
    AssistantMessage,
    Message,
    UserMessage,
)
from mcp.server.fastmcp.resources import FileResource
from mcp.types import PromptMessage
from pydantic import AnyUrl

from mcp_agent.mcp import mime_utils, resource_utils
from mcp_agent.mcp.prompts.prompt_constants import (
    ASSISTANT_DELIMITER as DEFAULT_ASSISTANT_DELIMITER,
)
from mcp_agent.mcp.prompts.prompt_constants import (
    RESOURCE_DELIMITER as DEFAULT_RESOURCE_DELIMITER,
)
from mcp_agent.mcp.prompts.prompt_constants import (
    USER_DELIMITER as DEFAULT_USER_DELIMITER,
)
from mcp_agent.mcp.prompts.prompt_load import create_messages_with_resources
from mcp_agent.mcp.prompts.prompt_template import (
    PromptMetadata,
    PromptTemplateLoader,
)

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("prompt_server")

# Create FastMCP server
mcp = FastMCP("Prompt Server")


def convert_to_fastmcp_messages(prompt_messages: List[PromptMessage]) -> List[Message]:
    """
    Convert PromptMessage objects from prompt_load to FastMCP Message objects.
    This adapter prevents double-wrapping of messages.

    Args:
        prompt_messages: List of PromptMessage objects from prompt_load

    Returns:
        List of FastMCP Message objects
    """
    result = []

    for msg in prompt_messages:
        if msg.role == "user":
            result.append(UserMessage(content=msg.content))
        elif msg.role == "assistant":
            result.append(AssistantMessage(content=msg.content))
        else:
            logger.warning(f"Unknown message role: {msg.role}, defaulting to user")
            result.append(UserMessage(content=msg.content))

    return result


class PromptConfig(PromptMetadata):
    """Configuration for the prompt server"""

    prompt_files: List[Path] = []
    user_delimiter: str = DEFAULT_USER_DELIMITER
    assistant_delimiter: str = DEFAULT_ASSISTANT_DELIMITER
    resource_delimiter: str = DEFAULT_RESOURCE_DELIMITER
    http_timeout: float = 10.0
    transport: str = "stdio"
    port: int = 8000
    host: str = "0.0.0.0"


# We'll maintain registries of all exposed resources and prompts
exposed_resources: Dict[str, Path] = {}
prompt_registry: Dict[str, PromptMetadata] = {}


# Define a single type for prompt handlers to avoid mypy issues
PromptHandler = Callable[..., Awaitable[List[Message]]]


# Type for resource handler
ResourceHandler = Callable[[], Awaitable[str | bytes]]


def create_resource_handler(resource_path: Path, mime_type: str) -> ResourceHandler:
    """Create a resource handler function for the given resource"""

    async def get_resource() -> str | bytes:
        is_binary = mime_utils.is_binary_content(mime_type)

        if is_binary:
            # For binary files, read in binary mode and base64 encode
            with open(resource_path, "rb") as f:
                return f.read()
        else:
            # For text files, read as utf-8 text
            with open(resource_path, "r", encoding="utf-8") as f:
                return f.read()

    return get_resource


def get_delimiter_config(
    config: Optional[PromptConfig] = None, file_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Get delimiter configuration, falling back to defaults if config is None"""
    # Set defaults
    config_values = {
        "user_delimiter": DEFAULT_USER_DELIMITER,
        "assistant_delimiter": DEFAULT_ASSISTANT_DELIMITER,
        "resource_delimiter": DEFAULT_RESOURCE_DELIMITER,
        "prompt_files": [file_path] if file_path else [],
    }

    # Override with config values if available
    if config is not None:
        config_values["user_delimiter"] = config.user_delimiter
        config_values["assistant_delimiter"] = config.assistant_delimiter
        config_values["resource_delimiter"] = config.resource_delimiter
        config_values["prompt_files"] = config.prompt_files

    return config_values


def register_prompt(file_path: Path, config: Optional[PromptConfig] = None) -> None:
    """Register a prompt file"""
    try:
        # Check if it's a JSON file for ultra-minimal path
        file_str = str(file_path).lower()
        if file_str.endswith(".json"):
            # Simple JSON handling - just load and register directly
            from mcp.server.fastmcp.prompts.base import Prompt, PromptArgument

            from mcp_agent.mcp.prompts.prompt_load import load_prompt

            # Create metadata with minimal information
            metadata = PromptMetadata(
                name=file_path.stem,
                description=f"JSON prompt: {file_path.stem}",
                template_variables=set(),
                resource_paths=[],  # Skip resource handling
                file_path=file_path,
            )

            # Ensure unique name
            prompt_name = metadata.name
            if prompt_name in prompt_registry:
                base_name = prompt_name
                suffix = 1
                while prompt_name in prompt_registry:
                    prompt_name = f"{base_name}_{suffix}"
                    suffix += 1
                metadata.name = prompt_name

            prompt_registry[metadata.name] = metadata

            # Create a simple handler that directly loads the JSON file each time
            async def json_prompt_handler():
                # Load the messages from the JSON file
                messages = load_prompt(file_path)
                # Convert to FastMCP format
                return convert_to_fastmcp_messages(messages)

            # Register directly with MCP
            prompt = Prompt(
                name=metadata.name,
                description=metadata.description,
                arguments=[],  # No arguments for JSON prompts
                fn=json_prompt_handler,
            )
            mcp._prompt_manager.add_prompt(prompt)

            logger.info(f"Registered JSON prompt: {metadata.name} ({file_path})")
            return  # Early return - we're done with JSON files

        # For non-JSON files, continue with the standard approach
        # Get delimiter configuration
        config_values = get_delimiter_config(config, file_path)

        # Use standard template loader for text files
        loader = PromptTemplateLoader(
            {
                config_values["user_delimiter"]: "user",
                config_values["assistant_delimiter"]: "assistant",
                config_values["resource_delimiter"]: "resource",
            }
        )

        # Get metadata and load the template
        metadata = loader.get_metadata(file_path)
        template = loader.load_from_file(file_path)

        # Ensure unique name
        prompt_name = metadata.name
        if prompt_name in prompt_registry:
            base_name = prompt_name
            suffix = 1
            while prompt_name in prompt_registry:
                prompt_name = f"{base_name}_{suffix}"
                suffix += 1
            metadata.name = prompt_name

        prompt_registry[metadata.name] = metadata
        logger.info(f"Registered prompt: {metadata.name} ({file_path})")

        # Create and register prompt handler
        template_vars = list(metadata.template_variables)

        from mcp.server.fastmcp.prompts.base import Prompt, PromptArgument

        # For prompts with variables, create arguments list for FastMCP
        if template_vars:
            # Create a function with properly typed parameters
            async def template_handler_with_vars(**kwargs):
                # Extract template variables from kwargs
                context = {
                    var: kwargs.get(var) for var in template_vars if var in kwargs
                }

                # Check for missing variables
                missing_vars = [var for var in template_vars if var not in context]
                if missing_vars:
                    raise ValueError(
                        f"Missing required template variables: {', '.join(missing_vars)}"
                    )

                # Apply template and create messages
                content_sections = template.apply_substitutions(context)
                prompt_messages = create_messages_with_resources(
                    content_sections, config_values["prompt_files"]
                )
                return convert_to_fastmcp_messages(prompt_messages)

            # Create a Prompt directly
            arguments = [
                PromptArgument(
                    name=var, description=f"Template variable: {var}", required=True
                )
                for var in template_vars
            ]

            # Create and add the prompt directly to the prompt manager
            prompt = Prompt(
                name=metadata.name,
                description=metadata.description,
                arguments=arguments,
                fn=template_handler_with_vars,
            )
            mcp._prompt_manager.add_prompt(prompt)
        else:
            # Create a simple prompt without variables
            async def template_handler_without_vars() -> list[Message]:
                content_sections = template.content_sections
                prompt_messages = create_messages_with_resources(
                    content_sections, config_values["prompt_files"]
                )
                return convert_to_fastmcp_messages(prompt_messages)

            # Create a Prompt object directly instead of using the decorator
            prompt = Prompt(
                name=metadata.name,
                description=metadata.description,
                arguments=[],
                fn=template_handler_without_vars,
            )
            mcp._prompt_manager.add_prompt(prompt)

        # Register any referenced resources in the prompt
        for resource_path in metadata.resource_paths:
            if not resource_path.startswith(("http://", "https://")):
                # It's a local resource
                resource_file = file_path.parent / resource_path
                if resource_file.exists():
                    resource_id = f"resource://fast-agent/{resource_file.name}"

                    # Register the resource if not already registered
                    if resource_id not in exposed_resources:
                        exposed_resources[resource_id] = resource_file
                        mime_type = mime_utils.guess_mime_type(str(resource_file))

                        mcp.add_resource(
                            FileResource(
                                uri=AnyUrl(resource_id),
                                path=resource_file,
                                mime_type=mime_type,
                                is_binary=mime_utils.is_binary_content(mime_type),
                            )
                        )

                        logger.info(
                            f"Registered resource: {resource_id} ({resource_file})"
                        )
    except Exception as e:
        logger.error(f"Error registering prompt {file_path}: {e}", exc_info=True)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="FastMCP Prompt Server")
    parser.add_argument(
        "prompt_files", nargs="+", type=str, help="Prompt files to serve"
    )
    parser.add_argument(
        "--user-delimiter",
        type=str,
        default="---USER",
        help="Delimiter for user messages (default: ---USER)",
    )
    parser.add_argument(
        "--assistant-delimiter",
        type=str,
        default="---ASSISTANT",
        help="Delimiter for assistant messages (default: ---ASSISTANT)",
    )
    parser.add_argument(
        "--resource-delimiter",
        type=str,
        default="---RESOURCE",
        help="Delimiter for resource references (default: ---RESOURCE)",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=10.0,
        help="Timeout for HTTP requests in seconds (default: 10.0)",
    )
    parser.add_argument(
        "--transport",
        type=str,
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="Transport to use (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to use for SSE transport (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to for SSE transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--test", type=str, help="Test a specific prompt without starting the server"
    )

    return parser.parse_args()


def initialize_config(args) -> PromptConfig:
    """Initialize configuration from command line arguments"""
    # Resolve file paths
    prompt_files = []
    for file_path in args.prompt_files:
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            continue
        prompt_files.append(path.resolve())

    if not prompt_files:
        logger.error("No valid prompt files specified")
        raise ValueError("No valid prompt files specified")

    # Initialize configuration
    return PromptConfig(
        name="prompt_server",
        description="FastMCP Prompt Server",
        template_variables=set(),
        resource_paths=[],
        file_path=Path(__file__),
        prompt_files=prompt_files,
        user_delimiter=args.user_delimiter,
        assistant_delimiter=args.assistant_delimiter,
        resource_delimiter=args.resource_delimiter,
        http_timeout=args.http_timeout,
        transport=args.transport,
        port=args.port,
        host=args.host,
    )


async def register_file_resource_handler(config: PromptConfig) -> None:
    """Register the general file resource handler"""

    @mcp.resource("file://{path}")
    async def get_file_resource(path: str):
        """Read a file from the given path."""
        try:
            # Find the file, checking relative paths first
            file_path = resource_utils.find_resource_file(path, config.prompt_files)
            if file_path is None:
                # If not found as relative path, try absolute path
                file_path = Path(path)
                if not file_path.exists():
                    raise FileNotFoundError(f"Resource file not found: {path}")

            mime_type = mime_utils.guess_mime_type(str(file_path))
            is_binary = mime_utils.is_binary_content(mime_type)

            if is_binary:
                # For binary files, read as binary and base64 encode
                with open(file_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            else:
                # For text files, read as text with UTF-8 encoding
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            # Log the error and re-raise
            logger.error(f"Error accessing resource at '{path}': {e}")
            raise


async def test_prompt(prompt_name: str, config: PromptConfig) -> int:
    """Test a prompt and print its details"""
    if prompt_name not in prompt_registry:
        logger.error(f"Test prompt not found: {prompt_name}")
        return 1

    # Get delimiter configuration with reasonable defaults
    config_values = get_delimiter_config(config)

    metadata = prompt_registry[prompt_name]
    print(f"\nTesting prompt: {prompt_name}")
    print(f"Description: {metadata.description}")
    print(f"Template variables: {', '.join(metadata.template_variables)}")

    # Load and print the template
    loader = PromptTemplateLoader(
        {
            config_values["user_delimiter"]: "user",
            config_values["assistant_delimiter"]: "assistant",
            config_values["resource_delimiter"]: "resource",
        }
    )
    template = loader.load_from_file(metadata.file_path)

    # Print each content section
    print("\nContent sections:")
    for i, section in enumerate(template.content_sections):
        print(f"\n[{i + 1}] Role: {section.role}")
        print(f"Content: {section.text}")
        if section.resources:
            print(f"Resources: {', '.join(section.resources)}")

    # If there are template variables, test with dummy values
    if metadata.template_variables:
        print("\nTemplate substitution test:")
        test_context = {var: f"[TEST-{var}]" for var in metadata.template_variables}
        applied = template.apply_substitutions(test_context)

        for i, section in enumerate(applied):
            print(f"\n[{i + 1}] Role: {section.role}")
            print(f"Content with substitutions: {section.text}")
            if section.resources:
                print(f"Resources with substitutions: {', '.join(section.resources)}")

    return 0


async def async_main() -> int:
    """Run the FastMCP server (async version)"""
    # Parse command line arguments
    args = parse_args()

    try:
        # Initialize configuration
        config = initialize_config(args)
    except ValueError as e:
        logger.error(str(e))
        return 1

    # Register resource handlers
    await register_file_resource_handler(config)

    # Register all prompts
    for file_path in config.prompt_files:
        register_prompt(file_path, config)

    # Print startup info
    logger.info("Starting prompt server")
    logger.info(f"Registered {len(prompt_registry)} prompts")
    logger.info(f"Registered {len(exposed_resources)} resources")
    logger.info(
        f"Using delimiters: {config.user_delimiter}, {config.assistant_delimiter}, {config.resource_delimiter}"
    )

    # If a test prompt was specified, print it and exit
    if args.test:
        return await test_prompt(args.test, config)

    # Start the server with the specified transport
    if config.transport == "sse":  # sse
        # Set the host and port in settings before running the server
        mcp.settings.host = config.host
        mcp.settings.port = config.port
        logger.info(f"Starting SSE server on {config.host}:{config.port}")
        await mcp.run_sse_async()
    elif config.transport == "http":
        mcp.settings.host = config.host
        mcp.settings.port = config.port
        logger.info(f"Starting SSE server on {config.host}:{config.port}")
        await mcp.run_streamable_http_async()
    elif config.transport == "stdio":
        await mcp.run_stdio_async()
    else:
        logger.error(f"Unknown transport: {config.transport}")
        return 1
    return 0


def main() -> int:
    """Run the FastMCP server"""
    try:
        return asyncio.run(async_main())
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
    except Exception as e:
        logger.error(f"\nError: {e}", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
