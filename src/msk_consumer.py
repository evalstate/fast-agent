#!/usr/bin/env python3

import asyncio
import json
import logging
import os
from aiokafka import AIOKafkaConsumer
from aiokafka.abc import AbstractTokenProvider
from aws_msk_iam_sasl_signer import MSKAuthTokenProvider
import ssl
import signal
import sys

def load_msk_config(config_dict=None):
    """
    Load MSK configuration from dictionary or environment variables
    
    Args:
        config_dict: Optional dictionary with MSK configuration
        
    Returns:
        Dictionary with MSK configuration
    """
    if config_dict:
        return {
            'bootstrap_servers': config_dict.get('bootstrap_servers', BOOTSTRAP_SERVERS),
            'topic_name': config_dict.get('topic_name', TOPIC_NAME),
            'aws_region': config_dict.get('aws_region', AWS_REGION),
            'consumer_group': config_dict.get('consumer_group', CONSUMER_GROUP),
        }
    
    return {
        'bootstrap_servers': BOOTSTRAP_SERVERS,
        'topic_name': TOPIC_NAME,
        'aws_region': AWS_REGION,
        'consumer_group': CONSUMER_GROUP,
    }

# Configuration - Load from environment variables with defaults
BOOTSTRAP_SERVERS = os.environ.get('MSK_BOOTSTRAP_SERVERS', 
    'b-3-public.commandhive.aewd11.c4.kafka.ap-south-1.amazonaws.com:9198,b-1-public.commandhive.aewd11.c4.kafka.ap-south-1.amazonaws.com:9198,b-2-public.commandhive.aewd11.c4.kafka.ap-south-1.amazonaws.com:9198').split(',')
TOPIC_NAME = os.environ.get('MSK_TOPIC_NAME', 'my-test-topic')
AWS_REGION = os.environ.get('AWS_REGION', 'ap-south-1')
CONSUMER_GROUP = os.environ.get('MSK_CONSUMER_GROUP', 'my-consumer-group')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variable to handle graceful shutdown
shutdown_event = asyncio.Event()

def create_ssl_context():
    """
    Create SSL context for secure connection to MSK
    """
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_context.options |= ssl.OP_NO_SSLv2
    ssl_context.options |= ssl.OP_NO_SSLv3
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    ssl_context.load_default_certs()
    return ssl_context

class AWSTokenProvider(AbstractTokenProvider):
    """
    AWS MSK IAM token provider for authentication
    """
    def __init__(self, region=AWS_REGION):
        self.region = region
    
    async def token(self):
        """
        Generate and return AWS MSK IAM token
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._generate_token)
    
    def _generate_token(self):
        """
        Generate token using MSKAuthTokenProvider
        """
        try:
            token, _ = MSKAuthTokenProvider.generate_auth_token(self.region)
            return token
        except Exception as e:
            logger.error(f"Failed to generate auth token: {e}")
            raise

async def create_consumer(bootstrap_servers, topic_name, consumer_group):
    """
    Create and return an async Kafka consumer with IAM authentication
    """
    try:
        tp = AWSTokenProvider()
        consumer = AIOKafkaConsumer(
            topic_name,
            bootstrap_servers=bootstrap_servers,
            group_id=consumer_group,
            security_protocol='SASL_SSL',
            ssl_context=create_ssl_context(),
            sasl_mechanism='OAUTHBEARER',
            sasl_oauth_token_provider=tp,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')) if m else None,
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            auto_offset_reset='earliest',  # Start from beginning if no committed offset
            enable_auto_commit=True,
            auto_commit_interval_ms=1000,
            client_id='msk_consumer',
            api_version="0.11.5",
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000
        )
        
        await consumer.start()
        logger.info(f"Async Kafka consumer created successfully for topic '{topic_name}'!")
        logger.info(f"Consumer group: {consumer_group}")
        return consumer
        
    except Exception as e:
        logger.error(f"Failed to create consumer: {str(e)}")
        return None

async def consume_messages(consumer):
    """
    Consume messages from the Kafka topic
    """
    message_count = 0
    
    try:
        logger.info("Starting to consume messages... (Press Ctrl+C to stop)")
        logger.info("=" * 60)
        
        async for msg in consumer:
            if shutdown_event.is_set():
                break
                
            message_count += 1
            
            # Display message information
            logger.info(f"📨 Message #{message_count}")
            logger.info(f"   Topic: {msg.topic}")
            logger.info(f"   Partition: {msg.partition}")
            logger.info(f"   Offset: {msg.offset}")
            logger.info(f"   Timestamp: {msg.timestamp}")
            logger.info(f"   Key: {msg.key}")
            
            # Pretty print the message value
            if msg.value:
                logger.info("   Value:")
                try:
                    # Pretty print JSON
                    formatted_json = json.dumps(msg.value, indent=4)
                    for line in formatted_json.split('\n'):
                        logger.info(f"     {line}")
                except Exception as e:
                    logger.info(f"     {msg.value}")
                    logger.warning(f"     (Could not format as JSON: {e})")
            else:
                logger.info("   Value: None")
            
            logger.info("-" * 60)
            
    except asyncio.CancelledError:
        logger.info("Consumer cancelled")
    except Exception as e:
        logger.error(f"Error consuming messages: {str(e)}")
    finally:
        logger.info(f"Total messages consumed: {message_count}")

def signal_handler(signum, frame):
    """
    Handle shutdown signals gracefully
    """
    logger.info(f"\nReceived signal {signum}. Shutting down gracefully...")
    shutdown_event.set()

async def main():
    """
    Main async function
    """
    logger.info("AWS MSK Async Message Consumer")
    logger.info("=" * 35)
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create consumer
    logger.info(f"Creating consumer for topic '{TOPIC_NAME}'...")
    consumer = await create_consumer(BOOTSTRAP_SERVERS, TOPIC_NAME, CONSUMER_GROUP)
    if not consumer:
        logger.error("Failed to create consumer. Exiting.")
        return
    
    try:
        # Get topic metadata
        partitions = consumer.assignment()
        if partitions:
            logger.info(f"Assigned partitions: {[tp.partition for tp in partitions]}")
        
        # Start consuming messages
        consume_task = asyncio.create_task(consume_messages(consumer))
        
        # Wait for shutdown signal
        await shutdown_event.wait()
        
        # Cancel the consume task
        consume_task.cancel()
        try:
            await consume_task
        except asyncio.CancelledError:
            pass
        
    finally:
        # Clean up
        logger.info("Stopping consumer...")
        await consumer.stop()
        logger.info("Consumer stopped.")
    
    logger.info("Consumer script completed!")

def run_async_main():
    """
    Run the async main function
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    run_async_main()