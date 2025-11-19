import unittest
from typing import Any, Dict, List

from fast_agent.llm.memory import SimpleMemory


def create_message(role: str, text: str, turn: int = 0) -> Dict[str, Any]:
    """Create a realistic message dict."""
    return {"role": role, "content": [{"type": "text", "text": f"{text} (turn {turn})"}]}


def create_tool_response_message(tool_result: str, turn: int = 0) -> Dict[str, Any]:
    """Create a tool response message."""
    return {
        "role": "user",
        "content": [{"type": "text", "text": f"Tool result: {tool_result} (turn {turn})"}],
    }


def has_cache_control(message: Dict[str, Any]) -> bool:
    """Check if a message has cache control."""
    if not isinstance(message, dict) or "content" not in message:
        return False
    content_list = message["content"]
    if not isinstance(content_list, list):
        return False
    for content_block in content_list:
        if isinstance(content_block, dict) and "cache_control" in content_block:
            return True
    return False


def count_cache_blocks(messages: List[Dict[str, Any]]) -> int:
    """Count total cache blocks in message array."""
    return sum(1 for msg in messages if has_cache_control(msg))


class TestCacheWalkingRealMessages(unittest.TestCase):
    """Test cache walking algorithm with real message arrays."""

    def setUp(self):
        """Set up test environment."""
        self.memory = SimpleMemory()

    def test_basic_walking_algorithm(self):
        """Test basic cache walking with real messages."""
        # Add prompt messages first
        prompt_messages = [
            create_message("user", "System prompt part 1"),
            create_message("assistant", "System prompt part 2"),
        ]
        self.memory.extend(prompt_messages, is_prompt=True)

        # Add conversation messages gradually
        conversation_messages = []
        cache_events = []

        for turn in range(1, 15):
            # Add user message
            user_msg = create_message("user", "User question", turn)
            conversation_messages.append(user_msg)
            self.memory.extend([user_msg])

            # Check for cache updates
            if self.memory.should_apply_conversation_cache():
                full_messages = self.memory.get(include_completion_history=True)
                updates = self.memory.get_conversation_cache_updates()

                # Apply updates to real message array
                self.memory.remove_cache_control_from_messages(full_messages, updates["remove"])
                applied = self.memory.add_cache_control_to_messages(full_messages, updates["add"])
                if applied > 0:
                    self.memory.apply_conversation_cache_updates(updates)
                    cache_events.append(
                        {
                            "turn": turn,
                            "total_messages": len(full_messages),
                            "updates": updates,
                            "cache_blocks": count_cache_blocks(full_messages),
                        }
                    )

            # Add assistant response
            assistant_msg = create_message("assistant", "Assistant response", turn)
            conversation_messages.append(assistant_msg)
            self.memory.extend([assistant_msg])

        # Verify cache walking behavior
        self.assertGreater(len(cache_events), 0, "Should have cache events")

        # Check final state
        final_messages = self.memory.get(include_completion_history=True)
        final_cache_count = count_cache_blocks(final_messages)
        self.assertLessEqual(final_cache_count, 2, "Should not exceed max cache blocks")

        print("\n=== Basic Walking Algorithm Test ===")
        print("Total turns: 14")
        print(f"Final messages: {len(final_messages)}")
        print(f"Cache events: {len(cache_events)}")
        print(f"Final cache blocks: {final_cache_count}")
        for event in cache_events:
            print(f"Turn {event['turn']}: {event['updates']}")

    def test_tool_calls_included_in_walking(self):
        """Test that tool calls are included in the walking algorithm."""
        # Add some conversation with tool calls
        messages_data = [
            ("user", "Question 1"),
            ("assistant", "Let me use a tool"),
            ("user", "Tool result: data"),  # Tool response
            ("assistant", "Based on tool result: answer 1"),
            ("user", "Question 2"),
            ("assistant", "Direct answer 2"),
            ("user", "Question 3"),
            ("assistant", "Let me use another tool"),
            ("user", "Tool result: more data"),  # Tool response
            ("assistant", "Based on tool result: answer 3"),
            ("user", "Question 4"),
            ("assistant", "Answer 4"),
        ]

        for i, (role, text) in enumerate(messages_data):
            msg = create_message(role, text, turn=i // 2 + 1)
            self.memory.extend([msg])

        # Apply cache walking
        if self.memory.should_apply_conversation_cache():
            full_messages = self.memory.get(include_completion_history=True)
            updates = self.memory.get_conversation_cache_updates()
            applied = self.memory.add_cache_control_to_messages(full_messages, updates["add"])
            if applied > 0:
                self.memory.apply_conversation_cache_updates(updates)

        # Verify tool messages are included in counting
        full_messages = self.memory.get(include_completion_history=True)
        cache_positions = [i for i, msg in enumerate(full_messages) if has_cache_control(msg)]

        self.assertGreater(len(cache_positions), 0, "Should have cache positions")

        # Check what types of messages got cached
        cached_messages = [full_messages[pos] for pos in cache_positions]
        cached_roles = [msg["role"] for msg in cached_messages]

        print("\n=== Tool Calls Inclusion Test ===")
        print(f"Total messages: {len(full_messages)}")
        print(f"Cache positions: {cache_positions}")
        print(f"Cached message roles: {cached_roles}")

        # Verify we cached meaningful messages (could be user or assistant)
        self.assertTrue(all(role in ["user", "assistant"] for role in cached_roles))

    def test_prompt_and_conversation_cache_coexistence(self):
        """Test that prompt caching and conversation caching work together."""
        # Add template/prompt messages with cache control (simulating "prompt" mode)
        prompt_messages = [
            create_message("user", "Template instruction"),
            create_message("assistant", "Template response"),
        ]
        self.memory.extend(prompt_messages, is_prompt=True)

        # Manually add cache control to simulate prompt caching
        full_messages = self.memory.get(include_completion_history=True)
        if full_messages:
            # Add cache to last prompt message
            last_prompt = full_messages[-1]
            if "content" in last_prompt and last_prompt["content"]:
                last_prompt["content"][0]["cache_control"] = {"type": "ephemeral"}

        # Add conversation messages
        for turn in range(1, 10):
            user_msg = create_message("user", f"Question {turn}", turn)
            assistant_msg = create_message("assistant", f"Answer {turn}", turn)
            self.memory.extend([user_msg, assistant_msg])

        # Apply conversation caching
        if self.memory.should_apply_conversation_cache():
            full_messages = self.memory.get(include_completion_history=True)
            updates = self.memory.get_conversation_cache_updates()
            applied = self.memory.add_cache_control_to_messages(full_messages, updates["add"])
            if applied > 0:
                self.memory.apply_conversation_cache_updates(updates)

        # Verify both prompt and conversation caches exist
        full_messages = self.memory.get(include_completion_history=True)
        cache_count = count_cache_blocks(full_messages)

        # Should have prompt cache + conversation cache(s)
        self.assertGreaterEqual(cache_count, 2, "Should have both prompt and conversation caches")
        self.assertLessEqual(cache_count, 4, "Should not exceed Anthropic's 4 cache block limit")

        print("\n=== Prompt + Conversation Cache Test ===")
        print(f"Total messages: {len(full_messages)}")
        print(f"Total cache blocks: {cache_count}")

        # Show which messages have cache
        for i, msg in enumerate(full_messages):
            if has_cache_control(msg):
                role = msg.get("role", "unknown")
                text = msg.get("content", [{}])[0].get("text", "")[:50]
                print(f"  Position {i} ({role}): {text}...")

    def test_cache_block_limit_safety(self):
        """Test that we never exceed the 4 cache block limit."""
        # Add prompt messages
        prompt_messages = [create_message("user", "Prompt")]
        self.memory.extend(prompt_messages, is_prompt=True)

        # Add lots of conversation to trigger multiple cache walks
        for turn in range(1, 25):  # 25 turns = 50 messages
            user_msg = create_message("user", f"Question {turn}", turn)
            assistant_msg = create_message("assistant", f"Answer {turn}", turn)
            self.memory.extend([user_msg, assistant_msg])

            # Apply cache updates after each turn
            if self.memory.should_apply_conversation_cache():
                full_messages = self.memory.get(include_completion_history=True)
                updates = self.memory.get_conversation_cache_updates()

                self.memory.remove_cache_control_from_messages(full_messages, updates["remove"])
                applied = self.memory.add_cache_control_to_messages(full_messages, updates["add"])
                if applied > 0:
                    self.memory.apply_conversation_cache_updates(updates)

                # Verify cache block limit
                cache_count = count_cache_blocks(full_messages)
                self.assertLessEqual(
                    cache_count, 4, f"Turn {turn}: Cache blocks ({cache_count}) exceed limit"
                )

        # Final verification
        final_messages = self.memory.get(include_completion_history=True)
        final_cache_count = count_cache_blocks(final_messages)

        print("\n=== Cache Block Limit Safety Test ===")
        print("Total turns: 24")
        print(f"Total messages: {len(final_messages)}")
        print(f"Final cache blocks: {final_cache_count}")
        print("Max allowed: 4")

        self.assertLessEqual(final_cache_count, 4, "Must never exceed 4 cache blocks")

    def test_different_walk_distances(self):
        """Test different walk distances for optimization."""
        test_distances = [4, 6, 8]
        results = {}

        for distance in test_distances:
            # Create fresh memory with specific walk distance
            memory = SimpleMemory()
            memory.cache_walk_distance = distance
            memory.max_conversation_cache_blocks = 2

            # Add 20 conversation messages
            for i in range(20):
                msg = create_message("user" if i % 2 == 0 else "assistant", f"Message {i}")
                memory.extend([msg])

            # Apply caching
            if memory.should_apply_conversation_cache():
                full_messages = memory.get(include_completion_history=True)
                updates = memory.get_conversation_cache_updates()
                applied = memory.add_cache_control_to_messages(full_messages, updates["add"])
                if applied > 0:
                    memory.apply_conversation_cache_updates(updates)

            # Record results
            full_messages = memory.get(include_completion_history=True)
            cache_positions = [i for i, msg in enumerate(full_messages) if has_cache_control(msg)]
            results[distance] = cache_positions

        print("\n=== Walk Distance Comparison ===")
        for distance, positions in results.items():
            print(f"Distance {distance}: Cache at positions {positions}")

        # Verify different distances produce different caching patterns
        self.assertNotEqual(
            results[4], results[8], "Different walk distances should produce different patterns"
        )


if __name__ == "__main__":
    unittest.main()
