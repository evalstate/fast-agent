"""
Redis listener for the logging system that publishes log events to Redis PubSub channels.
This allows all logger.info (and other log level) messages to be captured and published to Redis.
"""

import asyncio
import json
from typing import Dict, Any, Optional

from mcp_agent.logging.events import Event, EventFilter, EventType
from mcp_agent.logging.listeners import FilteredListener
from mcp_agent.logging.json_serializer import JSONSerializer
from mcp_agent.mcp.pubsub import get_pubsub_manager, PubSubChannel


class RedisLoggerListener(FilteredListener):
    """
    Listener that forwards log events to a PubSub channel.
    Supports Redis, Kafka, or in-memory backends.
    This allows all logger.info calls to be published for 
    subscription by other services.
    """

    def __init__(
        self,
        channel_name: str,
        event_filter: Optional[EventFilter] = None,
        backend: str = "memory",
        backend_config: Optional[Dict[str, Any]] = None,
        use_redis: bool = False,
        redis_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the logger listener.
        
        Args:
            channel_name: Name of the PubSub channel to publish events to
            event_filter: Optional filter to limit which events are published
            backend: Backend type - "memory", "redis", or "kafka"
            backend_config: Backend-specific configuration dictionary
            use_redis: (Deprecated) Whether to use Redis for pub/sub messaging
            redis_config: (Deprecated) Redis configuration dictionary
        """
        super().__init__(event_filter=event_filter)
        self.channel_name = channel_name
        self.serializer = JSONSerializer()
        
        # Initialize PubSub manager and channel
        self.pubsub_manager = get_pubsub_manager(
            backend=backend, 
            backend_config=backend_config,
            use_redis=use_redis, 
            redis_config=redis_config
        )
        self.channel: Optional[PubSubChannel] = None

    async def start(self) -> None:
        """Initialize the PubSub channel and verify MSK topic creation if using MSK backend."""
        print(f"🔧 Starting logger listener for channel: {self.channel_name}")
        self.channel = self.pubsub_manager.get_or_create_channel(self.channel_name)
        
        # If using MSK backend, ensure the channel is properly started and topic is created
        if hasattr(self.channel, 'start') and hasattr(self.channel, 'topic'):
            try:
                await self.channel.start()
                print(f"✅ MSK topic verified/created: {self.channel.topic}")
            except Exception as e:
                print(f"⚠️  Failed to start MSK channel: {e}")
                
        print(f"✅ Logger listener channel ready: {self.channel_name}")

    async def stop(self) -> None:
        """Clean up resources if needed."""
        pass  # PubSub manager handles channel cleanup

    async def handle_matched_event(self, event: Event) -> None:
        """
        Process a log event by publishing it to the Redis PubSub channel.
        
        Args:
            event: The log event to process
        """
        if not self.channel:
            return
        
        # Format the event as a dictionary for JSON serialization
        event_data = {
            "timestamp": event.timestamp.isoformat(),
            "type": event.type,
            "name": event.name,
            "namespace": event.namespace,
            "message": event.message,
            "data": self.serializer(event.data),
            "trace_id": event.trace_id,
            "span_id": event.span_id,
            "context": event.context.dict() if event.context else None,
        }
        
        # Publish the event to the PubSub channel
        await self.channel.publish(event_data)