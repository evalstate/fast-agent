"""
FastMCP prompt server.

Loads prompts from files and exposes them over stdio or HTTP.
"""

import argparse
import base64
import keyword
import logging
import re
import sys
from collections.abc import Awaitable, Callable, Sequence
from inspect import Parameter, Signature
from pathlib import Path
from typing import Any, cast

from fastmcp import FastMCP
from fastmcp.prompts import Message, Prompt, PromptArgument, PromptResult
from fastmcp.prompts.function_prompt import FunctionPrompt
from fastmcp.resources import FileResource
from mcp.types import BlobResourceContents, EmbeddedResource, ImageContent, PromptMessage
from pydantic import AnyUrl, Field

from fast_agent.mcp import mime_utils, resource_utils
from fast_agent.mcp.prompts.prompt_constants import (
    ASSISTANT_DELIMITER as DEFAULT_ASSISTANT_DELIMITER,
)
from fast_agent.mcp.prompts.prompt_constants import (
    RESOURCE_DELIMITER as DEFAULT_RESOURCE_DELIMITER,
)
from fast_agent.mcp.prompts.prompt_constants import (
    USER_DELIMITER as DEFAULT_USER_DELIMITER,
)
from fast_agent.mcp.prompts.prompt_load import (
    create_messages_with_resources,
    load_prompt,
    prompt_file_template_variables,
)
from fast_agent.mcp.prompts.prompt_template import PromptMetadata, PromptTemplateLoader
from fast_agent.types import PromptMessageExtended
from fast_agent.utils.async_utils import run_sync
from fast_agent.utils.text import strip_casefold

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("prompt_server")

mcp = FastMCP("Prompt Server")


def convert_to_fastmcp_messages(
    prompt_messages: Sequence[PromptMessage | PromptMessageExtended],
) -> list[Message]:
    """Convert MCP prompt messages into FastMCP prompt messages."""
    result: list[Message] = []
    for msg in prompt_messages:
        flat_messages = msg.from_multipart() if isinstance(msg, PromptMessageExtended) else [msg]
        for flat_msg in flat_messages:
            role = flat_msg.role if flat_msg.role in {"user", "assistant"} else "user"
            content = flat_msg.content
            if isinstance(content, ImageContent):
                content = EmbeddedResource(
                    type="resource",
                    resource=BlobResourceContents(
                        uri=AnyUrl("resource://fast-agent/prompt-image"),
                        blob=content.data,
                        mimeType=content.mimeType,
                    ),
                )
            result.append(Message(content=content, role=role))
    return result


class PromptConfig(PromptMetadata):
    """Configuration for the prompt server."""

    prompt_files: list[Path] = Field(default_factory=list)
    user_delimiter: str = DEFAULT_USER_DELIMITER
    assistant_delimiter: str = DEFAULT_ASSISTANT_DELIMITER
    resource_delimiter: str = DEFAULT_RESOURCE_DELIMITER
    http_timeout: float = 10.0
    transport: str = "stdio"
    port: int = 8000
    host: str = "127.0.0.1"


exposed_resources: dict[str, Path] = {}
prompt_registry: dict[str, PromptMetadata] = {}

PromptHandler = Callable[..., Awaitable[list[Message]]]


class _TemplateFunctionPrompt(FunctionPrompt):
    """Function prompt that exposes raw template variable names via argument aliases."""

    argument_name_map: dict[str, str] = Field(default_factory=dict, exclude=True)
    internal_arguments: list[PromptArgument] = Field(default_factory=list, exclude=True)

    async def render(self, arguments: dict[str, Any] | None = None) -> PromptResult:
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise ValueError("Prompt arguments must be a dict.")

        translated_arguments: dict[str, Any] = {}
        for name, value in arguments.items():
            translated_name = self.argument_name_map.get(name, name)
            if (
                translated_name in translated_arguments
                and translated_arguments[translated_name] != value
            ):
                raise ValueError(f"Duplicate values provided for argument '{name}'.")
            translated_arguments[translated_name] = value

        original_arguments = self.arguments
        try:
            self.arguments = self.internal_arguments or original_arguments
            return await super().render(translated_arguments)
        finally:
            self.arguments = original_arguments


def get_delimiter_config(
    config: PromptConfig | None = None, file_path: Path | None = None
) -> dict[str, Any]:
    """Get delimiter configuration, falling back to defaults if config is None."""
    values = {
        "user_delimiter": DEFAULT_USER_DELIMITER,
        "assistant_delimiter": DEFAULT_ASSISTANT_DELIMITER,
        "resource_delimiter": DEFAULT_RESOURCE_DELIMITER,
        "prompt_files": [file_path] if file_path else [],
    }
    if config is not None:
        values["user_delimiter"] = config.user_delimiter
        values["assistant_delimiter"] = config.assistant_delimiter
        values["resource_delimiter"] = config.resource_delimiter
        values["prompt_files"] = config.prompt_files
    return values


def _unique_prompt_name(metadata: PromptMetadata) -> PromptMetadata:
    prompt_name = metadata.name
    if prompt_name not in prompt_registry:
        return metadata

    base_name = prompt_name
    suffix = 1
    while prompt_name in prompt_registry:
        prompt_name = f"{base_name}_{suffix}"
        suffix += 1
    metadata.name = prompt_name
    return metadata


def _prompt_messages_for_template(
    template: Any,
    prompt_files: list[Path],
    context: dict[str, str] | None = None,
) -> list[Message]:
    sections = template.apply_substitutions(context or {}) if context else template.content_sections
    prompt_messages = create_messages_with_resources(sections, prompt_files)
    return convert_to_fastmcp_messages(prompt_messages)


def _build_dynamic_prompt_handler(
    *,
    metadata: PromptMetadata,
    template_vars: list[str],
    template_var_aliases: dict[str, str],
    render_context: Callable[[dict[str, str]], list[Message]],
) -> PromptHandler:
    async def handler(**kwargs: str) -> list[Message]:
        missing = [var for var in template_vars if template_var_aliases[var] not in kwargs]
        if missing:
            raise ValueError(f"Missing required template variables: {', '.join(missing)}")
        context = {var: kwargs[template_var_aliases[var]] for var in template_vars}
        return render_context(context)

    handler.__name__ = metadata.name
    handler.__annotations__ = {template_var_aliases[var]: str for var in template_vars}
    handler.__annotations__["return"] = list[Message]
    cast("Any", handler).__signature__ = Signature(
        parameters=[
            Parameter(
                name=template_var_aliases[var],
                kind=Parameter.POSITIONAL_OR_KEYWORD,
                annotation=str,
            )
            for var in template_vars
        ],
        return_annotation=list[Message],
    )
    return handler


def _sanitize_template_var_name(name: str) -> str:
    sanitized = re.sub(r"\W+", "_", name).strip("_")
    if not sanitized:
        sanitized = "template_var"
    if sanitized[0].isdigit():
        sanitized = f"template_{sanitized}"
    if keyword.iskeyword(sanitized):
        sanitized = f"{sanitized}_value"
    if not sanitized.isidentifier():
        sanitized = "template_var"
    return sanitized


def _template_var_aliases(template_vars: Sequence[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    used_names: set[str] = set()
    for index, var_name in enumerate(template_vars, start=1):
        candidate = var_name if var_name.isidentifier() and not keyword.iskeyword(var_name) else ""
        if not candidate:
            candidate = _sanitize_template_var_name(var_name)
        if not candidate or candidate in used_names:
            candidate = f"{candidate or 'template_var'}_{index}"
        while (
            candidate in used_names or keyword.iskeyword(candidate) or not candidate.isidentifier()
        ):
            candidate = f"{candidate}_{index}"
        aliases[var_name] = candidate
        used_names.add(candidate)
    return aliases


def _build_dynamic_prompt(
    *,
    metadata: PromptMetadata,
    handler: PromptHandler,
    template_var_aliases: dict[str, str],
) -> Prompt:
    prompt = cast(
        "_TemplateFunctionPrompt",
        _TemplateFunctionPrompt.from_function(
            handler,
            name=metadata.name,
            description=metadata.description,
        ),
    )
    prompt.argument_name_map = dict(template_var_aliases)
    prompt.internal_arguments = list(prompt.arguments or [])
    safe_arguments = {arg.name: arg for arg in prompt.internal_arguments}
    prompt.arguments = [
        PromptArgument(
            name=raw_name,
            description=safe_arguments[alias].description if alias in safe_arguments else None,
            required=safe_arguments[alias].required if alias in safe_arguments else True,
        )
        for raw_name, alias in template_var_aliases.items()
    ]
    return prompt


def _register_prompt_handler(
    *,
    metadata: PromptMetadata,
    handler: PromptHandler | None = None,
    prompt: Prompt | None = None,
) -> None:
    if prompt is None:
        if handler is None:
            raise ValueError("Either handler or prompt must be provided.")
        prompt = Prompt.from_function(
            handler,
            name=metadata.name,
            description=metadata.description,
        )
    prompt_registry[metadata.name] = metadata
    mcp.add_prompt(prompt)


def _register_json_prompt(file_path: Path) -> PromptMetadata:
    template_vars = sorted(prompt_file_template_variables(file_path))
    metadata = _unique_prompt_name(
        PromptMetadata(
            name=file_path.stem,
            description=f"JSON prompt: {file_path.stem}",
            template_variables=set(template_vars),
            resource_paths=[],
            file_path=file_path,
        )
    )

    if template_vars:
        template_var_aliases = _template_var_aliases(template_vars)
        handler = _build_dynamic_prompt_handler(
            metadata=metadata,
            template_vars=template_vars,
            template_var_aliases=template_var_aliases,
            render_context=lambda context: convert_to_fastmcp_messages(
                load_prompt(file_path, arguments=context)
            ),
        )
        prompt = _build_dynamic_prompt(
            metadata=metadata,
            handler=handler,
            template_var_aliases=template_var_aliases,
        )
    else:

        async def handler() -> list[Message]:
            return convert_to_fastmcp_messages(load_prompt(file_path))

        prompt = None

    _register_prompt_handler(metadata=metadata, handler=handler, prompt=prompt)
    return metadata


def _template_loader_for_config(config_values: dict[str, Any]) -> PromptTemplateLoader:
    return PromptTemplateLoader(
        {
            config_values["user_delimiter"]: "user",
            config_values["assistant_delimiter"]: "assistant",
            config_values["resource_delimiter"]: "resource",
        }
    )


def _register_template_prompt(file_path: Path, config: PromptConfig | None) -> PromptMetadata:
    config_values = get_delimiter_config(config, file_path)
    loader = _template_loader_for_config(config_values)
    metadata = _unique_prompt_name(loader.get_metadata(file_path))
    template = loader.load_from_file(file_path)
    template_vars = sorted(metadata.template_variables)

    if template_vars:
        template_var_aliases = _template_var_aliases(template_vars)
        handler = _build_dynamic_prompt_handler(
            metadata=metadata,
            template_vars=template_vars,
            template_var_aliases=template_var_aliases,
            render_context=lambda context: _prompt_messages_for_template(
                template,
                config_values["prompt_files"],
                context,
            ),
        )
        prompt = _build_dynamic_prompt(
            metadata=metadata,
            handler=handler,
            template_var_aliases=template_var_aliases,
        )
    else:

        async def handler() -> list[Message]:
            return _prompt_messages_for_template(template, config_values["prompt_files"])

        prompt = None

    _register_prompt_handler(metadata=metadata, handler=handler, prompt=prompt)
    return metadata


def _register_prompt_resources(file_path: Path, resource_paths: Sequence[str]) -> None:
    for resource_path in resource_paths:
        if resource_path.startswith(("http://", "https://")):
            continue
        resource_file = file_path.parent / resource_path
        if not resource_file.exists():
            continue

        resource_id = f"resource://fast-agent/{resource_file.name}"
        if resource_id in exposed_resources:
            continue

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
        logger.info(f"Registered resource: {resource_id} ({resource_file})")


def register_prompt(file_path: Path, config: PromptConfig | None = None) -> None:
    """Register a prompt file."""
    try:
        if strip_casefold(file_path.suffix) == ".json":
            metadata = _register_json_prompt(file_path)
            logger.info(f"Registered JSON prompt: {metadata.name} ({file_path})")
            return

        metadata = _register_template_prompt(file_path, config)
        logger.info(f"Registered prompt: {metadata.name} ({file_path})")
        _register_prompt_resources(file_path, metadata.resource_paths)
    except Exception as exc:
        logger.error(f"Error registering prompt {file_path}: {exc}", exc_info=True)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FastMCP Prompt Server")
    parser.add_argument("prompt_files", nargs="+", type=str, help="Prompt files to serve")
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
        choices=["stdio", "http"],
        default="stdio",
        help="Transport to use (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to use for HTTP transport (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to for HTTP transport (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--test", type=str, help="Test a specific prompt without starting the server"
    )
    return parser.parse_args()


def initialize_config(args) -> PromptConfig:
    """Initialize configuration from command line arguments."""
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
    """Register the general file resource handler."""

    @mcp.resource("file://{path}")
    async def get_file_resource(path: str):
        try:
            file_path = resource_utils.find_resource_file(path, config.prompt_files)
            if file_path is None:
                file_path = Path(path)
                if not file_path.exists():
                    raise FileNotFoundError(f"Resource file not found: {path}")

            mime_type = mime_utils.guess_mime_type(str(file_path))
            if mime_utils.is_binary_content(mime_type):
                with file_path.open("rb") as file:
                    return base64.b64encode(file.read()).decode("utf-8")
            with file_path.open("r", encoding="utf-8") as file:
                return file.read()
        except Exception as exc:
            logger.error(f"Error accessing resource at '{path}': {exc}")
            raise


async def test_prompt(prompt_name: str, config: PromptConfig) -> int:
    """Test a prompt and print its details."""
    if prompt_name not in prompt_registry:
        logger.error(f"Test prompt not found: {prompt_name}")
        return 1

    config_values = get_delimiter_config(config)
    metadata = prompt_registry[prompt_name]
    print(f"\nTesting prompt: {prompt_name}")
    print(f"Description: {metadata.description}")
    print(f"Template variables: {', '.join(metadata.template_variables)}")

    loader = PromptTemplateLoader(
        {
            config_values["user_delimiter"]: "user",
            config_values["assistant_delimiter"]: "assistant",
            config_values["resource_delimiter"]: "resource",
        }
    )
    template = loader.load_from_file(metadata.file_path)

    print("\nContent sections:")
    for index, section in enumerate(template.content_sections, start=1):
        print(f"\n[{index}] Role: {section.role}")
        print(f"Content: {section.text}")
        if section.resources:
            print(f"Resources: {', '.join(section.resources)}")

    if metadata.template_variables:
        print("\nTemplate substitution test:")
        test_context = {var: f"[TEST-{var}]" for var in metadata.template_variables}
        for index, section in enumerate(template.apply_substitutions(test_context), start=1):
            print(f"\n[{index}] Role: {section.role}")
            print(f"Content with substitutions: {section.text}")
            if section.resources:
                print(f"Resources with substitutions: {', '.join(section.resources)}")

    return 0


async def async_main() -> int:
    """Run the FastMCP server."""
    args = parse_args()

    try:
        config = initialize_config(args)
    except ValueError as exc:
        logger.error(str(exc))
        return 1

    await register_file_resource_handler(config)

    for file_path in config.prompt_files:
        register_prompt(file_path, config)

    logger.info("Starting prompt server")
    logger.info(f"Registered {len(prompt_registry)} prompts")
    logger.info(f"Registered {len(exposed_resources)} resources")
    logger.info(
        "Using delimiters: %s, %s, %s",
        config.user_delimiter,
        config.assistant_delimiter,
        config.resource_delimiter,
    )

    if args.test:
        return await test_prompt(args.test, config)

    if config.transport == "http":
        logger.info(f"Starting HTTP server on {config.host}:{config.port}")
        await mcp.run_http_async(transport="http", host=config.host, port=config.port)
        return 0
    if config.transport == "stdio":
        await mcp.run_stdio_async()
        return 0

    logger.error(f"Unknown transport: {config.transport}")
    return 1


def main() -> int:
    """Run the FastMCP server."""
    try:
        result = run_sync(async_main)
        return result if result is not None else 1
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user")
    except Exception as exc:
        logger.error(f"\nError: {exc}", exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
