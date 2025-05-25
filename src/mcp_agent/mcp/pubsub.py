from typing import Any, Callable, Dict, List, Optional, Set, Coroutine, Union
import asyncio
import json
import importlib.util
import sys


# Check if Redis is available
REDIS_AVAILABLE = importlib.util.find_spec("redis") is not None


class PubSubChannel:
    """
    A channel for publishing and subscribing to messages.
    Each channel has a unique identifier and can have multiple subscribers.
    """

    def __init__(self, channel_id: str) -> None:
        """
        Initialize a PubSub channel.
        
        Args:
            channel_id: Unique identifier for the channel
        """
        self.channel_id = channel_id
        self._subscribers: Set[Callable[[Any], None]] = set()
        # subscriber coroutines take Any and return None
        self._async_subscribers: Set[Callable[[Any], Coroutine[Any, Any, None]]] = set()
        self.history: List[Any] = []
        self.max_history_length = 100

    def subscribe(self, callback: Callable[[Any], None]) -> None:
        """
        Subscribe to the channel with a synchronous callback.
        
        Args:
            callback: Function to be called when a message is published
        """
        self._subscribers.add(callback)

    def unsubscribe(self, callback: Callable[[Any], None]) -> None:
        """
        Unsubscribe from the channel.
        
        Args:
            callback: The callback function to remove
        """
        self._subscribers.discard(callback)

    def subscribe_async(self, callback: Callable[[Any], Coroutine[Any, Any, None]]) -> None:
        """
        Subscribe to the channel with an asynchronous callback.
        
        Args:
            callback: Coroutine to be called when a message is published
        """
        self._async_subscribers.add(callback)

    def unsubscribe_async(self, callback: Callable[[Any], Coroutine[Any, Any, None]]) -> None:
        """
        Unsubscribe from the channel.
        
        Args:
            callback: The coroutine to remove
        """
        self._async_subscribers.discard(callback)

    async def publish(self, message: Any) -> None:
        """
        Publish a message to all subscribers.
        
        Args:
            message: The message to publish
        """
        # Add to history
        self.history.append(message)
        if len(self.history) > self.max_history_length:
            self.history = self.history[-self.max_history_length:]
        

        # Call synchronous subscribers
        for subscriber in list(self._subscribers):
            try:
                subscriber(message)
            except Exception as e:
                pass
        
        # Call asynchronous subscribers
        for subscriber in list(self._async_subscribers):
            try:
                await subscriber(message)
            except Exception as e:
                pass



class RedisPubSubChannel(PubSubChannel):
    """
    Redis-backed implementation of PubSubChannel.
    Uses Redis Pub/Sub for message distribution.
    """
    
    def __init__(self, channel_id: str, redis_client=None, redis_channel_prefix: str = "mcp_agent:") -> None:
        """
        Initialize a Redis-backed PubSub channel.
        
        Args:
            channel_id: Unique identifier for the channel
            redis_client: Redis client instance
            redis_channel_prefix: Prefix for Redis channel names
        """
        super().__init__(channel_id)
        self.redis_client = redis_client
        self.redis_channel = f"{redis_channel_prefix}{channel_id}"
        self._pubsub = None
        self._listener_task = None
        
        if redis_client:
            self._setup_redis_listener()
        
    async def _wait_for_subscription_ready(self, setup_task):
        """Wait for subscription to be ready before proceeding."""
        try:
            await setup_task
        except Exception as e:
            pass            
    def _setup_redis_listener(self) -> None:
        """Set up Redis subscription and message listener."""
        if not self.redis_client:
            return
            
        # For asyncio Redis, we need to await the subscribe operation
        async def setup_subscription():
            self._pubsub = self.redis_client.pubsub()
            await self._pubsub.subscribe(self.redis_channel)
        
        # Initialize subscription and wait for it to complete before continuing
        setup_task = asyncio.create_task(setup_subscription())
        # Wait for the setup to complete to ensure subscription is ready
        asyncio.create_task(self._wait_for_subscription_ready(setup_task))
        
        # Store a reference to self for the nested function to use
        channel_instance = self
        
        # Start listener in the background
        async def listener_loop() -> None:
            # Wait for pubsub to be initialized
            # Wait for pubsub to be properly initialized with subscription
            retry_count = 0
            max_retries = 10
            while (not hasattr(channel_instance, '_pubsub') or
                  not getattr(channel_instance._pubsub, 'subscribed', False)):
                retry_count += 1
                if retry_count > max_retries:
                    return
                    
                await asyncio.sleep(0.5)  # Longer sleep to give more time
                
            
            while True:
                try:
                    # For asyncio Redis, get_message is a coroutine and must be awaited
                    # Only log at debug level every 100 iterations to reduce noise
                    message = await channel_instance._pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
                    
                        
                    if message and message.get('type') == 'message':
                        data = message.get('data')
                        if isinstance(data, bytes):
                            data = data.decode('utf-8')
                        
                        
                        # Try to parse JSON, fall back to raw data if not JSON
                        try:
                            data = json.loads(data)
                        except json.JSONDecodeError:
                            pass
                            
                        # Call PubSubChannel's publish method to notify local subscribers
                        # Instead of using super() which doesn't work in nested functions
                        await PubSubChannel.publish(channel_instance, data)
                except Exception as e:
                    import traceback
                    
                await asyncio.sleep(0.01)  # Small delay to prevent CPU spike
                
        self._listener_task = asyncio.create_task(listener_loop())
    
    async def publish(self, message: Any) -> None:
        """
        Publish a message to Redis and local subscribers.
        
        Args:
            message: The message to publish
        """
        # Call superclass publish for local subscribers
        await super().publish(message)
        
        # Publish to Redis
        if self.redis_client:
            try:
                # Convert message to JSON if it's serializable
                if hasattr(message, 'to_dict'):
                    message_data = json.dumps(message.to_dict())
                elif isinstance(message, (dict, list, str, int, float, bool, type(None))):
                    message_data = json.dumps(message)
                else:
                    message_data = str(message)
                
                    
                # For asyncio Redis, publish is a coroutine and must be awaited
                await self.redis_client.publish(self.redis_channel, message_data)
            except Exception as e:
                pass

class PubSubManager:
    """
    Manager for PubSub channels.
    Handles creation, access, and cleanup of channels.
    """

    def __init__(self, use_redis: bool = False, redis_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the PubSub manager.
        
        Args:
            use_redis: Whether to use Redis for pub/sub messaging
            redis_config: Redis configuration dictionary (host, port, db, etc.)
        """
        self._channels: Dict[str, Union[PubSubChannel, RedisPubSubChannel]] = {}
        self.use_redis = use_redis and REDIS_AVAILABLE
        self.redis_client = None
        self.redis_channel_prefix = "mcp_agent:"
        
        if redis_config and 'channel_prefix' in redis_config:
            self.redis_channel_prefix = redis_config.pop('channel_prefix')
            
        if self.use_redis:
            try:
                import redis.asyncio as aioredis
                
                # Default Redis configuration
                redis_params = {
                    'host': 'localhost',
                    'port': 6379,
                    'db': 0,
                    'decode_responses': True
                }
                
                # Update with provided config
                if redis_config:
                    redis_params.update(redis_config)
                    
                self.redis_client = aioredis.Redis(**redis_params)
            except ImportError:
                self.use_redis = False
            except Exception as e:
                self.use_redis = False
        
        log_message = "Initialized PubSubManager"
        if self.use_redis:
            log_message += " with Redis backend"
        print(log_message)

    def get_or_create_channel(self, channel_id: str) -> Union[PubSubChannel, RedisPubSubChannel]:
        """
        Get an existing channel or create a new one.
        
        Args:
            channel_id: Unique identifier for the channel
            
        Returns:
            The requested PubSubChannel
        """
        if channel_id not in self._channels:
            if self.use_redis and self.redis_client:
                self._channels[channel_id] = RedisPubSubChannel(
                    channel_id, 
                    redis_client=self.redis_client,
                    redis_channel_prefix=self.redis_channel_prefix
                )
            else:
                self._channels[channel_id] = PubSubChannel(channel_id)
                
        return self._channels[channel_id]

    def get_channel(self, channel_id: str) -> Optional[Union[PubSubChannel, RedisPubSubChannel]]:
        """
        Get an existing channel.
        
        Args:
            channel_id: Unique identifier for the channel
            
        Returns:
            The requested PubSubChannel or None if it doesn't exist
        """
        return self._channels.get(channel_id)

    def remove_channel(self, channel_id: str) -> None:
        """
        Remove a channel.
        
        Args:
            channel_id: Unique identifier for the channel
        """
        if channel_id in self._channels:
            del self._channels[channel_id]

    def list_channels(self) -> List[str]:
        """
        List all channel IDs.
        
        Returns:
            List of channel IDs
        """
        return list(self._channels.keys())


# Default singleton instance of the PubSubManager
_pubsub_manager_instance = None


def get_pubsub_manager(use_redis: bool = False, redis_config: Optional[Dict[str, Any]] = None) -> PubSubManager:
    """
    Get or create the singleton PubSubManager instance.
    
    Args:
        use_redis: Whether to use Redis for pub/sub messaging
        redis_config: Redis configuration dictionary (host, port, db, etc.)
        
    Returns:
        The PubSubManager instance
    """
    global _pubsub_manager_instance
    
    if _pubsub_manager_instance is None:
        _pubsub_manager_instance = PubSubManager(use_redis=use_redis, redis_config=redis_config)
    
    return _pubsub_manager_instance