#!/bin/bash
# Clean and recreate dist folder
rm -rf dist
mkdir -p dist
# Build the package
uv build

# Extract version from the built wheel
VERSION=$(ls dist/fast_agent_mcp-*.whl | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | head -1)

# Create test folder
TEST_DIR="dist/test_install"
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

# Create virtual environment
uv venv .venv
source .venv/bin/activate

# Install the built package
uv pip install ../../dist/fast_agent_mcp-$VERSION-py3-none-any.whl

# Run the quickstart command
fast-agent quickstart workflow

# Check if workflows folder was created
if [ -d "workflow" ]; then
    echo "✅ Test successful: workflows folder created!"
else
    echo "❌ Test failed: workflows folder not created."
    exit 1
fi


# Run the quickstart command
fast-agent quickstart state-transfer
if [ -d "state-transfer" ]; then
    echo "✅ Test successful: state-transfer folder created!"
else
    echo "❌ Test failed: state-transfer folder not created."
    exit 1
fi

# Deactivate the virtual environment
deactivate

echo "Test completed successfully!"

