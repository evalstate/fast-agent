from typing import Any, Callable, Dict, List, Optional, Set, Coroutine, Union
import asyncio
import json
import importlib.util
import sys


# Check if Redis is available
REDIS_AVAILABLE = importlib.util.find_spec("redis") is not None

# Check if Kafka is available
KAFKA_AVAILABLE = importlib.util.find_spec("aiokafka") is not None
# print(f"IS KAFKA AVAILABLE??????????? {KAFKA_AVAILABLE}")

# Check if AWS MSK dependencies are available
MSK_AVAILABLE = (KAFKA_AVAILABLE and 
                importlib.util.find_spec("aws_msk_iam_sasl_signer") is not None and
                importlib.util.find_spec("boto3") is not None)

# print(f"IS MSK AVAILABLE??????????? {MSK_AVAILABLE}")

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
        # print(f"super channel streaming! {self.redis_channel}")
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


class KafkaPubSubChannel(PubSubChannel):
    """
    Kafka-backed implementation of PubSubChannel.
    Uses Kafka for message distribution with high throughput and persistence.
    """
    
    def __init__(self, channel_id: str, kafka_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize a Kafka-backed PubSub channel.
        
        Args:
            channel_id: Unique identifier for the channel (used as Kafka topic)
            kafka_config: Kafka configuration dictionary
        """
        super().__init__(channel_id)
        self.topic = f"mcp_agent_{channel_id}"
        self.kafka_config = kafka_config or {}
        
        # Default Kafka configuration
        self.producer_config = {
            'bootstrap_servers': self.kafka_config.get('bootstrap_servers', 'localhost:9092'),
            'value_serializer': lambda x: json.dumps(x).encode('utf-8') if isinstance(x, (dict, list)) else str(x).encode('utf-8'),
            **self.kafka_config.get('producer_config', {})
        }
        
        self.consumer_config = {
            'bootstrap_servers': self.kafka_config.get('bootstrap_servers', 'localhost:9092'),
            'group_id': f"mcp_agent_{channel_id}_group",
            'value_deserializer': lambda x: self._deserialize_message(x.decode('utf-8')),
            'auto_offset_reset': 'latest',
            **self.kafka_config.get('consumer_config', {})
        }
        
        self.producer = None
        self.consumer = None
        self._consumer_task = None
        self._is_consuming = False
        
    def _deserialize_message(self, message: str) -> Any:
        """Deserialize a message from JSON, falling back to string if not JSON."""
        try:
            return json.loads(message)
        except json.JSONDecodeError:
            return message
    
    async def _setup_kafka_consumer(self) -> None:
        """Set up Kafka consumer and start message listener."""
        if not KAFKA_AVAILABLE:
            return
            
        try:
            from aiokafka import AIOKafkaConsumer
            
            self.consumer = AIOKafkaConsumer(
                self.topic,
                **self.consumer_config
            )
            
            await self.consumer.start()
            self._is_consuming = True
            
            # Start consumer loop
            async def consumer_loop():
                try:
                    while self._is_consuming:
                        msg_batch = await self.consumer.getmany(timeout_ms=100, max_records=10)
                        for topic_partition, messages in msg_batch.items():
                            for message in messages:
                                try:
                                    # Call PubSubChannel's publish method to notify local subscribers
                                    await PubSubChannel.publish(self, message.value)
                                except Exception as e:
                                    pass
                except Exception as e:
                    pass
                    
            self._consumer_task = asyncio.create_task(consumer_loop())
            
        except ImportError:
            pass
        except Exception as e:
            pass
    
    async def start(self) -> None:
        """Initialize Kafka producer and consumer."""
        if not KAFKA_AVAILABLE:
            return
            
        try:
            from aiokafka import AIOKafkaProducer
            
            # Initialize producer
            self.producer = AIOKafkaProducer(**self.producer_config)
            await self.producer.start()
            
            # Initialize consumer
            await self._setup_kafka_consumer()
            
        except ImportError:
            pass
        except Exception as e:
            pass
    
    async def stop(self) -> None:
        """Clean up Kafka resources."""
        self._is_consuming = False
        
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        
        if self.consumer:
            try:
                await self.consumer.stop()
            except Exception:
                pass
                
        if self.producer:
            try:
                await self.producer.stop()
            except Exception:
                pass
    
    async def publish(self, message: Any) -> None:
        """
        Publish a message to Kafka and local subscribers.
        
        Args:
            message: The message to publish
        """
        # Call superclass publish for local subscribers
        await super().publish(message)
        
        # Publish to Kafka
        if self.producer and KAFKA_AVAILABLE:
            try:
                # Convert message to appropriate format
                if hasattr(message, 'to_dict'):
                    kafka_message = message.to_dict()
                elif isinstance(message, (dict, list, str, int, float, bool, type(None))):
                    kafka_message = message
                else:
                    kafka_message = str(message)
                
                await self.producer.send(self.topic, kafka_message)
                
            except Exception as e:
                pass


class MSKPubSubChannel(PubSubChannel):
    """
    AWS MSK (Managed Streaming for Kafka) implementation of PubSubChannel.
    Uses AWS MSK with IAM authentication for secure, managed Kafka messaging.
    """
    
    def __init__(self, channel_id: str, msk_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize an MSK-backed PubSub channel.
        
        Args:
            channel_id: Unique identifier for the channel (used as Kafka topic)
            msk_config: MSK configuration dictionary
        """
        super().__init__(channel_id)
        self.topic = f"mcp_agent_{channel_id}"
        self.msk_config = msk_config or {}
        # print("topic name")
        # print(self.topic)
        # print(msk_config)
        # Default MSK configuration
        self.bootstrap_servers = self.msk_config.get('bootstrap_servers', ['localhost:9092'])
        self.aws_region = self.msk_config.get('aws_region', 'ap-south-1')
        
        # Producer configuration
        self.producer_config = {
            'bootstrap_servers': self.bootstrap_servers,
            'security_protocol': 'SASL_SSL',
            'sasl_mechanism': 'OAUTHBEARER',
            'value_serializer': lambda x: json.dumps(x).encode('utf-8') if isinstance(x, (dict, list)) else str(x).encode('utf-8'),
            'acks': 'all',
            'client_id': f'mcp_agent_producer_{channel_id}',
            'api_version': "0.11.5",
            **self.msk_config.get('producer_config', {})
        }
        
        # Consumer configuration
        self.consumer_config = {
            'bootstrap_servers': self.bootstrap_servers,
            'group_id': f"mcp_agent_{channel_id}_group",
            'security_protocol': 'SASL_SSL',
            'sasl_mechanism': 'OAUTHBEARER',
            'value_deserializer': lambda x: self._deserialize_message(x.decode('utf-8')),
            'auto_offset_reset': 'latest',
            'enable_auto_commit': True,
            'client_id': f'mcp_agent_consumer_{channel_id}',
            'api_version': "0.11.5",
            **self.msk_config.get('consumer_config', {})
        }
        self.producer = None
        self.consumer = None
        self._consumer_task = None
        self._is_consuming = False
        self._ssl_context = None
        self._token_provider = None
        
    def _create_ssl_context(self):
        """Create SSL context for MSK connection."""
        if not self._ssl_context:
            import ssl
            self._ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            self._ssl_context.options |= ssl.OP_NO_SSLv2
            self._ssl_context.options |= ssl.OP_NO_SSLv3
            self._ssl_context.check_hostname = False
            self._ssl_context.verify_mode = ssl.CERT_NONE
            self._ssl_context.load_default_certs()
        return self._ssl_context
    
    def _create_token_provider(self):
        """Create AWS IAM token provider for MSK authentication."""
        if not self._token_provider and MSK_AVAILABLE:
            try:
                from aws_msk_iam_sasl_signer import MSKAuthTokenProvider
                from aiokafka.abc import AbstractTokenProvider
                
                class AWSTokenProvider(AbstractTokenProvider):
                    def __init__(self, region):
                        self.region = region
                    
                    async def token(self):
                        loop = asyncio.get_running_loop()
                        return await loop.run_in_executor(None, self._generate_token)
                    
                    def _generate_token(self):
                        try:
                            token, _ = MSKAuthTokenProvider.generate_auth_token(self.region)
                            return token
                        except Exception as e:
                            raise
                
                self._token_provider = AWSTokenProvider(self.aws_region)
            except ImportError:
                pass
        return self._token_provider
    
    def _deserialize_message(self, message: str) -> Any:
        """Deserialize a message from JSON, falling back to string if not JSON."""
        try:
            return json.loads(message)
        except json.JSONDecodeError:
            return message
    
    async def _create_topic_if_not_exists(self) -> bool:
        """Create the Kafka topic if it doesn't exist."""
        if not MSK_AVAILABLE:
            return False
            
        admin_client = None
        try:
            from aiokafka.admin import AIOKafkaAdminClient, NewTopic
            from aiokafka.errors import TopicAlreadyExistsError
            
            # Admin client configuration
            admin_config = {
                'bootstrap_servers': self.bootstrap_servers,
                'security_protocol': 'SASL_SSL',
                'sasl_mechanism': 'OAUTHBEARER',
                'ssl_context': self._create_ssl_context(),
                'client_id': f'mcp_agent_admin_{self.channel_id}',
                'api_version': "0.11.5"
            }
            
            token_provider = self._create_token_provider()
            if token_provider:
                admin_config['sasl_oauth_token_provider'] = token_provider
            
            admin_client = AIOKafkaAdminClient(**admin_config)
            await admin_client.start()
            
            # Create topic
            topic = NewTopic(
                name=self.topic,
                num_partitions=3,
                replication_factor=2
            )
            
            try:
                await admin_client.create_topics([topic])
                # print(f"Created MSK topic: {self.topic}")
                return True
            except TopicAlreadyExistsError:
                print(f"MSK topic already exists: {self.topic}")
                return True
            except Exception as e:
                print(f"Failed to create MSK topic {self.topic}: {e}")
                return False
                
        except Exception as e:
            print(f"Admin client error for topic {self.topic}: {e}")
            return False
        finally:
            if admin_client:
                try:
                    await admin_client.close()
                except Exception as e:
                    print(f"Error closing admin client: {e}")
    
    async def _setup_msk_consumer(self) -> None:
        """Set up MSK consumer and start message listener."""
        if not MSK_AVAILABLE:
            print(f"MSK not available for consumer setup on channel {self.channel_id}")
            return
            
        try:
            from aiokafka import AIOKafkaConsumer
            
            # print(f"Setting up MSK consumer for topic: {self.topic}")
            
            # Add MSK-specific configurations
            consumer_config = self.consumer_config.copy()
            consumer_config['ssl_context'] = self._create_ssl_context()
            
            token_provider = self._create_token_provider()
            if token_provider:
                consumer_config['sasl_oauth_token_provider'] = token_provider
            
            self.consumer = AIOKafkaConsumer(
                self.topic,
                **consumer_config
            )
            
            await self.consumer.start()
            self._is_consuming = True
            # print(f"MSK consumer started successfully for topic: {self.topic}")
            
            # Start consumer loop
            async def consumer_loop():
                try:
                    while self._is_consuming:
                        msg_batch = await self.consumer.getmany(timeout_ms=100, max_records=10)
                        for topic_partition, messages in msg_batch.items():
                            for message in messages:
                                try:
                                    # print(f"Received MSK message on topic {self.topic}: {message.value}")
                                    # Call PubSubChannel's publish method to notify local subscribers
                                    await PubSubChannel.publish(self, message.value)
                                except Exception as e:
                                    print(f"Error processing MSK message: {e}")
                except Exception as e:
                    print(f"MSK consumer loop error: {e}")
                    
            self._consumer_task = asyncio.create_task(consumer_loop())
            
        except ImportError as e:
            print(f"MSK consumer import error for channel {self.channel_id}: {e}")
        except Exception as e:
            print(f"MSK consumer setup error for channel {self.channel_id}: {e}")
            import traceback
            traceback.print_exc()
    
    async def start(self) -> None:
        """Initialize MSK producer and consumer."""
        if not MSK_AVAILABLE:
            print(f"MSK dependencies not available for channel {self.channel_id}")
            return
            
        try:
            from aiokafka import AIOKafkaProducer
            
            # print(f"Starting MSK channel for topic: {self.topic}")
            
            # Create topic first
            await self._create_topic_if_not_exists()
            
            # Add MSK-specific configurations
            producer_config = self.producer_config.copy()
            producer_config['ssl_context'] = self._create_ssl_context()
            
            token_provider = self._create_token_provider()
            if token_provider:
                producer_config['sasl_oauth_token_provider'] = token_provider
            
            # Initialize producer
            self.producer = AIOKafkaProducer(**producer_config)
            await self.producer.start()
            # print(f"MSK producer started successfully for topic: {self.topic}")
            
            # Initialize consumer
            await self._setup_msk_consumer()
            # print(f"MSK channel startup completed for topic: {self.topic}")
            
        except ImportError as e:
            print(f"MSK import error for channel {self.channel_id}: {e}")
        except Exception as e:
            print(f"MSK startup error for channel {self.channel_id}: {e}")
            import traceback
            traceback.print_exc()
    
    async def stop(self) -> None:
        """Clean up MSK resources."""
        self._is_consuming = False
        
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
        
        if self.consumer:
            try:
                await self.consumer.stop()
            except Exception:
                pass
                
        if self.producer:
            try:
                await self.producer.stop()
            except Exception:
                pass
    
    async def publish(self, message: Any) -> None:
        """
        Publish a message to MSK and local subscribers.
        
        Args:
            message: The message to publish
        """
        # Call superclass publish for local subscribers
        await super().publish(message)
        
        # Publish to MSK
        if self.producer and MSK_AVAILABLE:
            try:
                # Convert message to appropriate format
                if hasattr(message, 'to_dict'):
                    msk_message = message.to_dict()
                elif isinstance(message, (dict, list, str, int, float, bool, type(None))):
                    msk_message = message
                else:
                    msk_message = str(message)
                
                record_metadata = await self.producer.send_and_wait(self.topic, msk_message)
                # print(f"Published message to MSK topic {self.topic} at partition {record_metadata.partition}, offset {record_metadata.offset}")
                
            except Exception as e:
                # print(f"Failed to publish message to MSK topic {self.topic}: {e}")
                import traceback
                traceback.print_exc()
        else:
            if not self.producer:
                print(f"MSK producer not available for topic {self.topic}")
            if not MSK_AVAILABLE:
                print(f"MSK not available for topic {self.topic}")


class PubSubManager:
    """
    Manager for PubSub channels.
    Handles creation, access, and cleanup of channels.
    """

    def __init__(self, backend: str = "memory", backend_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the PubSub manager.
        
        Args:
            backend: Backend type - "memory", "redis", "kafka", or "msk"
            backend_config: Backend-specific configuration dictionary
        """
        self._channels: Dict[str, Union[PubSubChannel, RedisPubSubChannel, KafkaPubSubChannel, MSKPubSubChannel]] = {}
        self.backend = backend.lower()
        self.backend_config = backend_config or {}
        
        # Redis-specific setup
        self.use_redis = self.backend == "redis" and REDIS_AVAILABLE
        self.redis_client = None
        self.redis_channel_prefix = "mcp_agent:"
        
        # Kafka-specific setup  
        self.use_kafka = self.backend == "kafka" and KAFKA_AVAILABLE
        self.kafka_config = None
        
        # MSK-specific setup
        self.use_msk = self.backend == "msk" and MSK_AVAILABLE
        self.msk_config = None
        
        if self.use_redis:
            if self.backend_config and 'channel_prefix' in self.backend_config:
                self.redis_channel_prefix = self.backend_config.pop('channel_prefix')
                
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
                if self.backend_config:
                    redis_params.update(self.backend_config)
                    
                self.redis_client = aioredis.Redis(**redis_params)
            except ImportError:
                self.use_redis = False
                self.backend = "memory"
            except Exception as e:
                self.use_redis = False
                self.backend = "memory"
        
        elif self.use_kafka:
            self.kafka_config = self.backend_config
            
        elif self.use_msk:
            self.msk_config = self.backend_config
            
        # log_message = f"Initialized PubSubManager with {self.backend} backend"
        # print(log_message)

    def get_or_create_channel(self, channel_id: str) -> Union[PubSubChannel, RedisPubSubChannel, KafkaPubSubChannel, MSKPubSubChannel]:
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
            elif self.use_kafka:
                channel = KafkaPubSubChannel(channel_id, kafka_config=self.kafka_config)
                # Start the Kafka channel asynchronously
                asyncio.create_task(channel.start())
                self._channels[channel_id] = channel
            elif self.use_msk:
                channel = MSKPubSubChannel(channel_id, msk_config=self.msk_config)
                asyncio.create_task(channel.start())
                self._channels[channel_id] = channel
            else:
                self._channels[channel_id] = PubSubChannel(channel_id)
                
        return self._channels[channel_id]

    def get_channel(self, channel_id: str) -> Optional[Union[PubSubChannel, RedisPubSubChannel, KafkaPubSubChannel, MSKPubSubChannel]]:
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


def get_pubsub_manager(backend: str = "memory", backend_config: Optional[Dict[str, Any]] = None, 
                      use_redis: bool = False, redis_config: Optional[Dict[str, Any]] = None) -> PubSubManager:
    """
    Get or create the singleton PubSubManager instance.
    
    Args:
        backend: Backend type - "memory", "redis", "kafka", or "msk"
        backend_config: Backend-specific configuration dictionary
        use_redis: (Deprecated) Whether to use Redis for pub/sub messaging
        redis_config: (Deprecated) Redis configuration dictionary
        
    Returns:
        The PubSubManager instance
    """
    global _pubsub_manager_instance
    
    if _pubsub_manager_instance is None:
        # Handle backwards compatibility
        if use_redis and backend == "memory":
            backend = "redis"
            backend_config = redis_config
        
        # print(f"the backend from pubsub.py {backend}")
        _pubsub_manager_instance = PubSubManager(backend=backend, backend_config=backend_config)
        # print('pubsub manager instance')
        # print(_pubsub_manager_instance.__dict__)
    
    return _pubsub_manager_instance