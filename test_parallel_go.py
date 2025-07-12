#!/usr/bin/env python3
"""Test script for the parallel model feature in go.py"""

import subprocess
import sys

# Test case 1: Single model (should work as before)
print("Test 1: Single model")
result = subprocess.run(
    [sys.executable, "-m", "mcp_agent", "go", "--model=haiku", "--message=What is 2+2?"],
    capture_output=True,
    text=True
)
print(f"Exit code: {result.returncode}")
print(f"Output: {result.stdout}")
if result.stderr:
    print(f"Error: {result.stderr}")

print("\n" + "="*50 + "\n")

# Test case 2: Multiple models with comma delimiter
print("Test 2: Multiple models (haiku,sonnet)")
result = subprocess.run(
    [sys.executable, "-m", "mcp_agent", "go", "--model=haiku,sonnet", "--message=What is 2+2?"],
    capture_output=True,
    text=True
)
print(f"Exit code: {result.returncode}")
print(f"Output: {result.stdout}")
if result.stderr:
    print(f"Error: {result.stderr}")

print("\n" + "="*50 + "\n")

# Test case 3: Three models
print("Test 3: Three models (haiku,sonnet,gpt-4)")
result = subprocess.run(
    [sys.executable, "-m", "mcp_agent", "go", "--model=haiku,sonnet,gpt-4", "--message=What is the capital of France?"],
    capture_output=True,
    text=True
)
print(f"Exit code: {result.returncode}")
print(f"Output: {result.stdout}")
if result.stderr:
    print(f"Error: {result.stderr}")