#!/usr/bin/env python
"""
PubSub Client Example

This example demonstrates how to create a client application that connects to 
FastAgent using the PubSub system. The client can subscribe to agent channels
to receive messages and publish messages as user input.

Usage:
  python pubsub_client.py [agent_name]

If no agent name is provided, the client will attempt to list available channels
and let you choose one.
"""

import argparse
import asyncio
import json
import sys
import os
from typing import Any, Dict, List, Optional

sys.path.insert(0, "../../../..")  # Add the project root to the Python path

from mcp_agent.mcp.pubsub import get_pubsub_manager


class PubSubClient:
    """
    Client for interacting with FastAgent via PubSub.
    """

    def __init__(self, channel_id: Optional[str] = None, use_redis: bool = False, redis_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the PubSub client.
        
        Args:
            channel_id: Optional channel ID to connect to
            use_redis: Whether to use Redis for PubSub
            redis_config: Redis configuration
        """
        self.pubsub_manager = get_pubsub_manager(use_redis=use_redis, redis_config=redis_config)
        self.channel = None
        self.running = False
        
        if channel_id:
            self.connect(channel_id)
    
    def connect(self, channel_id: str) -> bool:
        """
        Connect to a specific channel.
        
        Args:
            channel_id: Channel ID to connect to
            
        Returns:
            True if connection was successful, False otherwise
        """
        self.channel = self.pubsub_manager.get_channel(channel_id)
        if not self.channel:
            print(f"Channel '{channel_id}' not found. Creating new channel.")
            self.channel = self.pubsub_manager.get_or_create_channel(channel_id)
        
        return self.channel is not None
    
    def list_channels(self) -> List[str]:
        """
        List all available channels.
        
        Returns:
            List of channel IDs
        """
        return self.pubsub_manager.list_channels()
    
    async def subscribe(self) -> None:
        """
        Subscribe to messages on the current channel.
        This method will print messages as they arrive.
        """
        if not self.channel:
            print("Not connected to any channel. Use connect() first.")
            return
        
        self.running = True
        
        async def message_handler(message: Any) -> None:
            """Handle incoming messages."""
            if hasattr(message, "to_dict"):
                message_dict = message.to_dict()
                self._print_formatted_message(message_dict)
            else:
                print(f"Received message: {message}")
        
        # Subscribe to the channel
        await self.channel.subscribe_async(message_handler)
        
        print(f"Subscribed to channel: {self.channel.channel_id}")
        print("Waiting for messages... (Press Ctrl+C to exit)")
        
        # Keep the client running
        while self.running:
            await asyncio.sleep(0.1)
    
    async def publish_user_message(self, message: str) -> None:
        """
        Publish a user message to the channel.
        
        Args:
            message: Message text to publish
        """
        if not self.channel:
            print("Not connected to any channel. Use connect() first.")
            return
        
        from mcp_agent.mcp.pubsub_formatter import PubSubFormatter
        
        # Format and publish the message
        formatted_message = PubSubFormatter.format_user_message(
            message=message,
            channel_id=self.channel.channel_id,
            metadata={"source": "pubsub_client"}
        )
        
        await self.channel.publish(formatted_message)
        print(f"Published user message: {message}")
    
    def _print_formatted_message(self, message: Dict[str, Any]) -> None:
        """
        Format and print a message based on its type.
        
        Args:
            message: Message dictionary to print
        """
        message_type = message.get("type", "unknown")
        content = message.get("content", "")
        
        if message_type == "assistant":
            print("\n=== ASSISTANT ===")
            print(content)
            print("================\n")
        
        elif message_type == "user":
            print("\n=== USER ===")
            print(content)
            print("============\n")
        
        elif message_type == "tool_call":
            tool_name = content.get("tool_name", "")
            tool_args = content.get("tool_args", {})
            
            print(f"\n=== TOOL CALL: {tool_name} ===")
            print(json.dumps(tool_args, indent=2))
            print("=======================\n")
        
        elif message_type == "tool_result":
            is_error = content.get("isError", False)
            result_content = content.get("content", "")
            
            if is_error:
                print("\n=== TOOL ERROR ===")
            else:
                print("\n=== TOOL RESULT ===")
            
            print(result_content)
            print("===================\n")
        
        else:
            print(f"\n=== {message_type.upper()} ===")
            print(json.dumps(content, indent=2))
            print("===================\n")


async def interactive_session(client: PubSubClient) -> None:
    """
    Run an interactive session with the client.
    
    Args:
        client: PubSubClient instance
    """
    # Start the subscription in the background
    subscription_task = asyncio.create_task(client.subscribe())
    
    try:
        while True:
            # Use asyncio.to_thread to avoid blocking the event loop
            user_input = await asyncio.to_thread(
                lambda: input("\nEnter a message (or 'exit' to quit): ")
            )
            
            if user_input.lower() in ("exit", "quit", "q"):
                client.running = False
                break
            
            await client.publish_user_message(user_input)
    
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
    
    finally:
        client.running = False
        await subscription_task


async def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="PubSub client for FastAgent")
    parser.add_argument("channel", nargs="?", help="Channel ID to connect to")
    parser.add_argument("--use-redis", action="store_true", help="Use Redis for PubSub")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--redis-db", type=int, default=0, help="Redis DB number")
    parser.add_argument("--channel-prefix", default="mcp_agent:", help="Redis channel prefix")
    args = parser.parse_args()
    
    # Configure Redis if enabled
    use_redis = args.use_redis
    redis_config = None
    
    if use_redis:
        redis_config = {
            "host": args.redis_host,
            "port": args.redis_port,
            "db": args.redis_db,
            "channel_prefix": args.channel_prefix
        }
        print(f"Using Redis at {args.redis_host}:{args.redis_port} with DB {args.redis_db}")
    
    client = PubSubClient(use_redis=use_redis, redis_config=redis_config)
    
    if args.channel:
        # Connect to the specified channel
        channel_id = args.channel
        if not channel_id.startswith("agent_"):
            channel_id = f"agent_{channel_id}"
        
        client.connect(channel_id)
    else:
        # List available channels
        channels = client.list_channels()
        if not channels:
            print("No channels available. Please specify a channel ID.")
            return
        
        print("Available channels:")
        for i, channel in enumerate(channels):
            print(f"{i+1}. {channel}")
        
        choice = input("Select a channel (number) or enter a new channel ID: ")
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(channels):
                client.connect(channels[idx])
            else:
                print("Invalid selection. Please enter a valid number.")
                return
        except ValueError:
            # Assume the user entered a channel ID
            channel_id = choice
            if not channel_id.startswith("agent_"):
                channel_id = f"agent_{channel_id}"
            
            client.connect(channel_id)
    
    if client.channel:
        await interactive_session(client)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")