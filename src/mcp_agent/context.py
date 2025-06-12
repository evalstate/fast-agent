"""
A central context object to store global state that is shared across the application.
"""

import asyncio
import concurrent.futures
import uuid
from typing import TYPE_CHECKING, Any, Optional, Union

from mcp import ServerSession
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor

# from opentelemetry.instrumentation.mcp import McpInstrumentor
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.propagate import set_global_textmap
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from pydantic import BaseModel, ConfigDict

from mcp_agent.config import Settings, get_settings
from mcp_agent.executor.executor import AsyncioExecutor, Executor
from mcp_agent.executor.task_registry import ActivityRegistry
from mcp_agent.logging.events import EventFilter
from mcp_agent.logging.logger import LoggingConfig, get_logger
from mcp_agent.logging.transport import create_transport
from mcp_agent.mcp_server_registry import ServerRegistry

if TYPE_CHECKING:
    from mcp_agent.executor.workflow_signal import SignalWaitCallback
    from mcp_agent.human_input.types import HumanInputCallback
else:
    # Runtime placeholders for the types
    HumanInputCallback = Any
    SignalWaitCallback = Any

logger = get_logger(__name__)


class Context(BaseModel):
    """
    Context that is passed around through the application.
    This is a global context that is shared across the application.
    """

    config: Optional[Settings] = None
    executor: Optional[Executor] = None
    human_input_handler: Optional[HumanInputCallback] = None
    signal_notification: Optional[SignalWaitCallback] = None
    upstream_session: Optional[ServerSession] = None  # TODO: saqadri - figure this out

    # Registries
    server_registry: Optional[ServerRegistry] = None
    task_registry: Optional[ActivityRegistry] = None

    tracer: trace.Tracer | None = None

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
    AnthropicInstrumentor().instrument()
    OpenAIInstrumentor().instrument()
    GoogleGenAiSdkInstrumentor().instrument()


#    McpInstrumentor().instrument()


async def configure_logger(config: "Settings") -> None:
    """
    Configure logging and tracing based on the application config.
    """
    event_filter: EventFilter = EventFilter(min_level=config.logger.level)
    logger.info(f"Configuring logger with level: {config.logger.level}")
    transport = create_transport(settings=config.logger, event_filter=event_filter)
    await LoggingConfig.configure(
        event_filter=event_filter,
        transport=transport,
        batch_size=config.logger.batch_size,
        flush_interval=config.logger.flush_interval,
        progress_display=config.logger.progress_display,
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
