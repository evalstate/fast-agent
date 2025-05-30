#!/usr/bin/env python3

import asyncio
import json
import time
import logging
import os
from aiokafka import AIOKafkaProducer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic
from aiokafka.errors import TopicAlreadyExistsError
import boto3
from aiokafka.abc import AbstractTokenProvider

from aws_msk_iam_sasl_signer import MSKAuthTokenProvider
import ssl

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
            'aws_access_key_id': config_dict.get('aws_access_key_id'),
            'aws_secret_access_key': config_dict.get('aws_secret_access_key'),
        }
    
    return {
        'bootstrap_servers': BOOTSTRAP_SERVERS,
        'topic_name': TOPIC_NAME,
        'aws_region': AWS_REGION,
        'aws_access_key_id': AWS_ACCESS_KEY_ID,
        'aws_secret_access_key': AWS_SECRET_ACCESS_KEY,
    }

# Configuration - Load from environment variables with defaults
BOOTSTRAP_SERVERS = os.environ.get('MSK_BOOTSTRAP_SERVERS', 
    'b-3-public.commandhive.aewd11.c4.kafka.ap-south-1.amazonaws.com:9198,b-1-public.commandhive.aewd11.c4.kafka.ap-south-1.amazonaws.com:9198,b-2-public.commandhive.aewd11.c4.kafka.ap-south-1.amazonaws.com:9198').split(',')
TOPIC_NAME = os.environ.get('MSK_TOPIC_NAME', 'my-test-topic')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.environ.get('AWS_REGION', 'ap-south-1')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

async def create_kafka_topic(bootstrap_servers, topic_name, num_partitions=3, replication_factor=2):
    """
    Create a Kafka topic if it doesn't exist using aiokafka admin client
    """
    admin_client = None
    try:
        tp = AWSTokenProvider()
        admin_client = AIOKafkaAdminClient(
            bootstrap_servers=bootstrap_servers,
            security_protocol='SASL_SSL',
            ssl_context=create_ssl_context(),
            sasl_mechanism='OAUTHBEARER',
            sasl_oauth_token_provider=tp,
            client_id='topic_creator',
            api_version="0.11.5"
        )
        
        await admin_client.start()
        
        topic = NewTopic(
            name=topic_name,
            num_partitions=num_partitions,
            replication_factor=replication_factor
        )
        
        try:
            await admin_client.create_topics([topic])
            logger.info(f"Topic '{topic_name}' created successfully!")
            return True
        except TopicAlreadyExistsError:
            logger.info(f"Topic '{topic_name}' already exists")
            return True
        except Exception as e:
            logger.error(f"Failed to create topic '{topic_name}': {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to connect to admin client: {str(e)}")
        return False
    finally:
        if admin_client:
            try:
                await admin_client.close()
            except Exception as e:
                logger.warning(f"Error closing admin client: {e}")

async def create_producer(bootstrap_servers):
    """
    Create and return an async Kafka producer with IAM authentication
    """
    try:
        tp = AWSTokenProvider()
        producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap_servers,
            security_protocol='SASL_SSL',
            ssl_context=create_ssl_context(),
            sasl_mechanism='OAUTHBEARER',
            sasl_oauth_token_provider=tp,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',  # Wait for all replicas to acknowledge
            client_id='msk_producer',
            api_version="0.11.5"
        )
        
        await producer.start()
        logger.info("Async Kafka producer created successfully!")
        return producer
        
    except Exception as e:
        logger.error(f"Failed to create producer: {str(e)}")
        return None

async def send_messages(producer, topic_name, num_messages=10):
    """
    Send sample messages to the Kafka topic asynchronously
    """
    try:
        for i in range(num_messages):
            message = {
                'id': i,
                'timestamp': time.time(),
                'message': f'Hello from AWS MSK - Async Message {i}',
                'data': {
                    'user_id': f'user_{i % 5}',
                    'action': 'sample_action',
                    'value': i * 10,
                    'environment': 'production'
                }
            }
            
            # Send message with key
            key = f'key_{i}'
            
            # Send and wait for acknowledgment
            record_metadata = await producer.send_and_wait(topic_name, key=key, value=message)
            
            logger.info(f"Message {i} sent to topic '{record_metadata.topic}' "
                       f"partition {record_metadata.partition} offset {record_metadata.offset}")
            
            # Small delay between messages
            await asyncio.sleep(0.5)
        
        logger.info(f"Successfully sent {num_messages} messages!")
        
    except Exception as e:
        logger.error(f"Error sending messages: {str(e)}")

async def verify_aws_credentials():
    """
    Verify AWS credentials by making a simple AWS call
    """
    try:
        # Use default credentials chain if explicit credentials not provided
        if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
            session = boto3.Session(
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION
            )
        else:
            # Use default credential chain (IAM roles, profiles, etc.)
            session = boto3.Session(region_name=AWS_REGION)
        
        # Try to list MSK clusters to verify credentials
        msk_client = session.client('kafka')
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: msk_client.list_clusters(MaxResults=1)
        )
        
        logger.info("AWS credentials verified successfully!")
        
        # Show available clusters if any
        if response.get('ClusterInfoList'):
            cluster_name = response['ClusterInfoList'][0].get('ClusterName', 'Unknown')
            logger.info(f"Found MSK cluster: {cluster_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"AWS credentials verification failed: {str(e)}")
        return False

async def test_connection(bootstrap_servers):
    """
    Test connection to MSK cluster
    """
    producer = None
    try:
        tp = AWSTokenProvider()
        producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap_servers,
            security_protocol='SASL_SSL',
            ssl_context=create_ssl_context(),
            sasl_mechanism='OAUTHBEARER',
            sasl_oauth_token_provider=tp,
            client_id='connection_test',
            api_version="0.11.5"
        )
        
        await producer.start()
        logger.info("Successfully connected to MSK cluster!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to connect to MSK cluster: {str(e)}")
        return False
    finally:
        if producer:
            try:
                await producer.stop()
            except Exception as e:
                logger.warning(f"Error stopping test producer: {e}")

async def main():
    """
    Main async function
    """
    logger.info("AWS MSK Async Topic Creator and Message Producer")
    logger.info("=" * 55)
    
    # Verify AWS credentials (optional)
    logger.info("Verifying AWS credentials...")
    if not await verify_aws_credentials():
        logger.warning("AWS credential verification failed. Proceeding anyway...")
    
    # Test connection to MSK
    logger.info("Testing connection to MSK cluster...")
    if not await test_connection(BOOTSTRAP_SERVERS):
        logger.error("Failed to connect to MSK cluster. Please check your configuration.")
        return
    
    # Create topic
    logger.info(f"Creating topic '{TOPIC_NAME}'...")
    if not await create_kafka_topic(BOOTSTRAP_SERVERS, TOPIC_NAME):
        logger.error("Failed to create topic. Exiting.")
        return
    await asyncio.sleep(20)
    # Create producer
    logger.info("Creating async Kafka producer...")
    producer = await create_producer(BOOTSTRAP_SERVERS)
    if not producer:
        logger.error("Failed to create producer. Exiting.")
        return
    
    try:
        # Option 1: Send JSON messages with keys
        logger.info(f"Sending JSON messages to topic '{TOPIC_NAME}'...")
        await send_messages(producer, TOPIC_NAME, num_messages=5)
                
        # Ensure all messages are sent
        await producer.flush()
        logger.info("All messages flushed successfully!")
        
    finally:
        # Clean up
        await producer.stop()
        logger.info("Producer stopped.")
    
    logger.info("Script completed successfully!")

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