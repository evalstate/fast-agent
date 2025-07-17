#!/usr/bin/env python3
"""Test to verify the agent decorator signature."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Try to import just the direct_decorators module
try:
    import inspect

    from mcp_agent.core.direct_decorators import agent

    print("Successfully imported agent decorator")

    # Check the signature
    sig = inspect.signature(agent)
    print(f"agent signature: {sig}")

    # Check the servers parameter specifically
    servers_param = sig.parameters["servers"]
    print(f"servers parameter: {servers_param}")
    print(f"servers annotation: {servers_param.annotation}")
    print(f"servers default: {servers_param.default}")

except Exception as e:
    print(f"Error: {e}")
    print("Let's try a different approach...")

    # Read the source directly
    with open("src/mcp_agent/core/direct_decorators.py", "r") as f:
        content = f.read()

    # Find the agent function definition
    lines = content.split("\n")
    in_agent_def = False
    for i, line in enumerate(lines):
        if "def agent(" in line:
            in_agent_def = True
            print(f"Found agent definition at line {i + 1}:")

        if in_agent_def:
            print(f"  {i + 1}: {line}")
            if "servers:" in line:
                print(f"  --> FOUND SERVERS PARAMETER: {line.strip()}")
            if line.strip().endswith("):") and "def agent(" not in line:
                break
