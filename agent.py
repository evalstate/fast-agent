
import asyncio
import json
import os
import ssl
import logging
from mcp_agent.core.fastagent import FastAgent
from dotenv import load_dotenv
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.abc import AbstractTokenProvider
from aws_msk_iam_sasl_signer import MSKAuthTokenProvider

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MSK Configuration
BOOTSTRAP_SERVERS = ['b-3-public.commandhive.aewd11.c4.kafka.ap-south-1.amazonaws.com:9198','b-1-public.commandhive.aewd11.c4.kafka.ap-south-1.amazonaws.com:9198', 'b-2-public.commandhive.aewd11.c4.kafka.ap-south-1.amazonaws.com:9198']
TOPIC_NAME = 'mcp_agent_queen'
AWS_REGION = os.environ.get('AWS_REGION', 'ap-south-1')
CONSUMER_GROUP = 'mcp_agent_consumer'

def create_ssl_context():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.options |= ssl.OP_NO_SSLv2 | ssl.OP_NO_SSLv3
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    ctx.load_default_certs()
    return ctx

class AWSTokenProvider(AbstractTokenProvider):
    def __init__(self, region=AWS_REGION):
        self.region = region

    async def token(self):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._generate_token)

    def _generate_token(self):
        token, _ = MSKAuthTokenProvider.generate_auth_token(self.region)
        return token

async def create_consumer():
    tp = AWSTokenProvider()
    consumer = AIOKafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=CONSUMER_GROUP,
        security_protocol='SASL_SSL',
        ssl_context=create_ssl_context(),
        sasl_mechanism='OAUTHBEARER',
        sasl_oauth_token_provider=tp,
        value_deserializer=lambda m: json.loads(m.decode('utf-8')) if m else None,
        auto_offset_reset='latest',
        enable_auto_commit=True,
        client_id='mcp_agent_consumer',
    )
    await consumer.start()
    logger.info(f"Connected to MSK topic '{TOPIC_NAME}'")
    return consumer

async def create_producer():
    tp = AWSTokenProvider()
    producer = AIOKafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        security_protocol='SASL_SSL',
        ssl_context=create_ssl_context(),
        sasl_mechanism='OAUTHBEARER',
        sasl_oauth_token_provider=tp,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all',
        client_id='mcp_agent_producer'
    )
    await producer.start()
    return producer

# Load environment variables
load_dotenv()

# Simple JSON config for MCP
simple_config = {
    "mcp": {
        "servers": {
            "fetch": {
                "name": "fetch",
                "description": "A server for fetching web content",
                "transport": "stdio",
                "command": "uvx",
                "args": ["mcp-server-fetch"]
            }
        }
    },
    "default_model": "claude-3-7-sonnet-latest",   
    "logger": {
        "level": "info",
        "type": "console"
    },
    "pubsub_enabled": True,
    "pubsub_config": {
        "backend": "msk",  # Options: "memory", "redis", "kafka", "msk"
        "channel_name": "queen",
        "msk": {
            "bootstrap_servers": [
                "b-3-public.commandhive.aewd11.c4.kafka.ap-south-1.amazonaws.com:9198",
                "b-1-public.commandhive.aewd11.c4.kafka.ap-south-1.amazonaws.com:9198", 
                "b-2-public.commandhive.aewd11.c4.kafka.ap-south-1.amazonaws.com:9198"
            ],
            "aws_region": "ap-south-1",
            "topic_prefix": "mcp_agent_",
            "security_protocol": "SASL_SSL",
            "sasl_mechanism": "OAUTHBEARER",
            "ssl_config": {
                "check_hostname": False,
                "verify_mode": "none"
            },
            "producer_config": {
                "acks": "all",
                "client_id": "mcp_agent_producer"
            },
            "consumer_config": {
                "auto_offset_reset": "latest",
                "enable_auto_commit": True,
                "client_id": "mcp_agent_consumer"
            }
        },
    },
    "anthropic": {
        "api_key": os.environ.get("CLAUDE_API_KEY", "") 
    }
}

# Create FastAgent instance
fast = FastAgent(
    name="simple_queen",
    json_config=simple_config,
    parse_cli_args=True  # Enable CLI argument parsing for model selection
)

@fast.agent(
    name="assistant",
    instruction="""You are a helpful AI assistant. You can:
    - Answer questions on any topic
    - Help with problem-solving
    - Fetch web content when needed using the fetch server
    - Have natural conversations
    
    Be friendly, helpful, and engaging in your responses.""",
    servers=["fetch"],  # Give access to web fetching capability
    model="claude-3-7-sonnet-latest"  # Default model, can be overridden with --model flag
)
async def main():
    """MCP Agent that listens to MSK pub/sub messages"""
    
    # Create Kafka consumer and producer
    consumer = await create_consumer()
    producer = await create_producer()
    
    async with fast.run() as agent:
        logger.info("🤖 MCP Agent is ready and listening for messages!")
        logger.info("Listening on MSK topic: %s", TOPIC_NAME)
        logger.info("Send messages using the producer script to interact with the agent")
        logger.info("-" * 50)
        
        try:
            # Listen for messages from MSK
            async for message in consumer:
                try:
                    if not message.value:
                        continue
                    
                    logger.info("Received message: %s", message.value)
                    
                    # Extract user content from message
                    user_content = None
                    if isinstance(message.value, dict):
                        if message.value.get('type') == 'user_message' and 'content' in message.value:
                            user_content = message.value['content']
                        elif message.value.get('type') == 'user' and 'content' in message.value:
                            user_content = message.value['content']
                    elif isinstance(message.value, str):
                        # Try to parse as JSON first
                        try:
                            data_obj = json.loads(message.value)
                            if data_obj.get('type') in ['user_message', 'user'] and 'content' in data_obj:
                                user_content = data_obj['content']
                            else:
                                user_content = message.value
                        except json.JSONDecodeError:
                            user_content = message.value
                    
                    if user_content:
                        logger.info("👤 User: %s", user_content)
                        
                        # Send to agent and get response
                        response = await agent.assistant(user_content)
                        logger.info("🤖 Assistant: %s", response)
                        
                        
                    
                except Exception as e:
                    logger.error("Error processing message: %s", e)
                    import traceback
                    traceback.print_exc()
                    
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
        except Exception as e:
            logger.error("Fatal error: %s", e)
        finally:
            await consumer.stop()
            await producer.stop()
            logger.info("Consumer and producer stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
        pass