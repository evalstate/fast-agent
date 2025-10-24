from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
# from opentelemetry.instrumentation.mcp import McpInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from pydantic import BaseModel, ConfigDict

from fast_agent.config import Settings, get_settings
from fast_agent.core.executor.executor import AsyncioExecutor, Executor
from fast_agent.core.executor.task_registry import ActivityRegistry
from fast_agent.core.logging.events import EventFilter, StreamingExclusionFilter
from fast_agent.core.logging.logger import LoggingConfig, get_logger
from fast_agent.core.logging.transport import create_transport
from fast_agent.mcp_server_registry import ServerRegistry
from fast_agent.skills import SkillRegistry

if TYPE_CHECKING:
    from fast_agent.core.executor.workflow_signal import SignalWaitCallback
    from fast_agent.mcp.mcp_connection_manager import MCPConnectionManager
else:
    # Runtime placeholders for the types
    SignalWaitCallback = Any
    MCPConnectionManager = Any

logger = get_logger(__name__)

"""
A central context object to store global state that is shared across the application.
"""


class Context(BaseModel):
    """
    Context that is passed around through the application.
    This is a global context that is shared across the application.
    """

    config: Optional[Settings] = None
    executor: Optional[Executor] = None
    human_input_handler: Optional[Any] = None
    signal_notification: Optional[SignalWaitCallback] = None

    # Registries
    server_registry: Optional[ServerRegistry] = None
    task_registry: Optional[ActivityRegistry] = None
    skill_registry: Optional[SkillRegistry] = None

    tracer: trace.Tracer | None = None
    _connection_manager: "MCPConnectionManager | None" = None

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,  # Tell Pydantic to defer type evaluation
    )


async def configure_otel(config: "Settings") -> None:
    """
    Configure OpenTelemetry based on the application config.
    """
    if not config.otel or not config.otel.enabled:
        return

    # Set up global textmap propagator first
    set_global_textmap(TraceContextTextMapPropagator())

    service_name = config.otel.service_name
    from importlib.metadata import version

    try:
        app_version = version("fast-agent-mcp")
    except:  # noqa: E722
        app_version = "unknown"

    resource = Resource.create(
        attributes={
            key: value
            for key, value in {
                "service.name": service_name,
                "service.instance.id": str(uuid.uuid4())[:6],
                "service.version": app_version,
            }.items()
            if value is not None
        }
    )

    # Create provider with resource
    tracer_provider = TracerProvider(resource=resource)

    # Add exporters based on config
    otlp_endpoint = config.otel.otlp_endpoint
    if otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))

        if config.otel.console_debug:
            tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    else:
        # Default to console exporter in development
        tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    # Set as global tracer provider
    trace.set_tracer_provider(tracer_provider)

    # Attempt to instrument optional SDKs if available; continue silently if missing
    try:
        from opentelemetry.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().instrument()
    except Exception:  # pragma: no cover - optional instrumentation
        pass

    try:
        from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor

        GoogleGenAiSdkInstrumentor().instrument()
    except Exception:  # pragma: no cover - optional instrumentation
        pass


#    McpInstrumentor().instrument()


async def configure_logger(config: "Settings") -> None:
    """
    Configure logging and tracing based on the application config.
    """
    settings = config.logger

    # Configure the standard Python logger used by LoggingListener so it respects settings.
    python_logger = logging.getLogger("fast_agent")
    python_logger.handlers.clear()
    python_logger.setLevel(settings.level.upper())
    python_logger.propagate = False

    transport = None
    if settings.type == "console":
        # Console mode: use the Python logger to emit to stdout and skip additional transport output
        handler = logging.StreamHandler()
        handler.setLevel(settings.level.upper())
        handler.setFormatter(logging.Formatter("%(message)s"))
        python_logger.addHandler(handler)
    else:
        # For all other modes, rely on transports (file/http/none) and keep the Python logger quiet
        python_logger.addHandler(logging.NullHandler())

    # Use StreamingExclusionFilter to prevent streaming events from flooding logs
    event_filter: EventFilter = StreamingExclusionFilter(min_level=settings.level)
    logger.info(f"Configuring logger with level: {settings.level}")
    if settings.type == "console":
        from fast_agent.core.logging.transport import NoOpTransport

        transport = NoOpTransport(event_filter=event_filter)
    else:
        transport = create_transport(settings=settings, event_filter=event_filter)
    await LoggingConfig.configure(
        event_filter=event_filter,
        transport=transport,
        batch_size=settings.batch_size,
        flush_interval=settings.flush_interval,
        progress_display=settings.progress_display,
    )


async def configure_usage_telemetry(_config: "Settings") -> None:
    """
    Configure usage telemetry based on the application config.
    TODO: saqadri - implement usage tracking
    """
    pass


async def configure_executor(config: "Settings"):
    """
    Configure the executor based on the application config.
    """
    return AsyncioExecutor()


async def initialize_context(
    config: Optional[Union["Settings", str]] = None, store_globally: bool = False
):
    """
    Initialize the global application context.
    """
    if config is None:
        config = get_settings()
    elif isinstance(config, str):
        config = get_settings(config_path=config)

    context = Context()
    context.config = config
    context.server_registry = ServerRegistry(config=config)

    skills_settings = getattr(config, "skills", None)
    override_directory = None
    if skills_settings and getattr(skills_settings, "directory", None):
        override_directory = Path(skills_settings.directory).expanduser()
    context.skill_registry = SkillRegistry(
        base_dir=Path.cwd(),
        override_directory=override_directory,
    )

    # Configure logging and telemetry
    await configure_otel(config)
    await configure_logger(config)
    await configure_usage_telemetry(config)

    # Configure the executor
    context.executor = await configure_executor(config)
    context.task_registry = ActivityRegistry()

    # Store the tracer in context if needed
    if config.otel:
        context.tracer = trace.get_tracer(config.otel.service_name)

    if store_globally:
        global _global_context
        _global_context = context

    return context


async def cleanup_context() -> None:
    """
    Cleanup the global application context.
    """

    # Shutdown logging and telemetry
    await LoggingConfig.shutdown()


_global_context: Context | None = None


def get_current_context() -> Context:
    """
    Synchronous initializer/getter for global application context.
    """
    global _global_context
    if _global_context is None:
        try:
            # Try to get the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new loop in a separate thread
                def run_async():
                    new_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(new_loop)
                    return new_loop.run_until_complete(initialize_context())

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    _global_context = pool.submit(run_async).result()
            else:
                _global_context = loop.run_until_complete(initialize_context())
        except RuntimeError:
            _global_context = asyncio.run(initialize_context())
    return _global_context


def get_current_config():
    """
    Get the current application config.

    Returns the context config if available, otherwise falls back to global settings.
    """
    return get_current_context().config or get_settings()
