#!/bin/bash
# Script to set up Git worktrees for Amazon Bedrock integration with fast-agent client
# This creates isolated development environments for different components of the implementation

set -e  # Exit immediately if a command exits with a non-zero status

# Store the current directory to return to it at the end
ORIGINAL_DIR=$(pwd)
echo "Setting up worktrees for Amazon Bedrock integration with fast-agent"
echo "------------------------------------------------"

# Create a parent branch for all Bedrock-related work
git checkout -b feature/bedrock-integration
git checkout main  # Go back to main to create the worktrees from it

# Define the worktree locations with their purposes
echo "Creating worktrees for isolated development environments..."

# 1. Core provider implementation - the main Bedrock provider code
git worktree add ../bedrock-provider feature/bedrock-provider
echo "✅ Created worktree for the core Bedrock provider in ../bedrock-provider"

# 2. Model configurations and mappings - for defining Bedrock models
git worktree add ../bedrock-models feature/bedrock-models
echo "✅ Created worktree for Bedrock model configurations in ../bedrock-models"

# 3. Configuration and authentication - for AWS credential handling
git worktree add ../bedrock-auth feature/bedrock-auth
echo "✅ Created worktree for AWS authentication in ../bedrock-auth"

# 4. Content conversion - for message format transformation
git worktree add ../bedrock-content feature/bedrock-content
echo "✅ Created worktree for content/message format conversion in ../bedrock-content"

# 5. Testing and examples - for implementation verification
git worktree add ../bedrock-testing feature/bedrock-testing
echo "✅ Created worktree for testing and examples in ../bedrock-testing"

echo "------------------------------------------------"
echo "Worktree setup complete! Here's how to use them:"
echo ""
echo "1. Provider Implementation (../bedrock-provider)"
echo "   Focus: Creating the main BedrockAugmentedLLM implementation"
echo "   Key files:"
echo "   - src/mcp_agent/llm/providers/augmented_llm_bedrock.py"
echo "   - src/mcp_agent/llm/provider_types.py (add BEDROCK enum)"
echo ""
echo "2. Model Configurations (../bedrock-models)"
echo "   Focus: Model definitions, mappings, and parameters"
echo "   Key files:"
echo "   - src/mcp_agent/llm/model_factory.py"
echo "   - Add Bedrock models to DEFAULT_PROVIDERS"
echo "   - Add any model aliases for Bedrock models"
echo ""
echo "3. AWS Authentication (../bedrock-auth)"
echo "   Focus: Implementing credential management for AWS"
echo "   Key files:"
echo "   - src/mcp_agent/llm/provider_key_manager.py"
echo "   - Handle AWS credentials and region configuration"
echo ""
echo "4. Content Conversion (../bedrock-content)"
echo "   Focus: Message formatting and content type handling"
echo "   Key files:"
echo "   - Create format converters specific to Bedrock if needed"
echo "   - Implement multipart content handling"
echo ""
echo "5. Tests and Examples (../bedrock-testing)"
echo "   Focus: Testing implementation and creating examples"
echo "   Key files:"
echo "   - tests/unit/mcp_agent/llm/providers/test_augmented_llm_bedrock.py"
echo "   - examples/bedrock/fastagent.config.yaml"
echo "   - examples/bedrock/agent.py"
echo ""
echo "------------------------------------------------"
echo "To launch Claude Code in each worktree:"
echo ""
echo "cd ../bedrock-provider && claude"
echo "cd ../bedrock-models && claude"
echo "cd ../bedrock-auth && claude"
echo "cd ../bedrock-content && claude"
echo "cd ../bedrock-testing && claude"
echo ""
echo "------------------------------------------------"
echo "Merging changes when complete:"
echo ""
echo "1. First merge each feature branch back to the main branch:"
echo "   git checkout feature/bedrock-integration"
echo "   git merge feature/bedrock-provider"
echo "   git merge feature/bedrock-models"
echo "   git merge feature/bedrock-auth"
echo "   git merge feature/bedrock-content"
echo "   git merge feature/bedrock-testing"
echo ""
echo "2. Create a pull request from feature/bedrock-integration to main"
echo ""
echo "3. Clean up worktrees when no longer needed:"
echo "   git worktree remove ../bedrock-provider"
echo "   git worktree remove ../bedrock-models"
echo "   git worktree remove ../bedrock-auth"
echo "   git worktree remove ../bedrock-content"
echo "   git worktree remove ../bedrock-testing"
echo "   git branch -D feature/bedrock-provider feature/bedrock-models feature/bedrock-auth feature/bedrock-content feature/bedrock-testing"
echo ""
echo "------------------------------------------------"

# Make script executable
chmod +x "$0"

# Return to the original directory
cd "$ORIGINAL_DIR"
echo "Setup complete! Execute this script to create the worktrees."