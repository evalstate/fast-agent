# Bedrock Integration Tests

This directory contains integration tests for the AWS Bedrock provider. These tests verify that the Bedrock provider works correctly with the fast-agent client.

## Test Categories

The tests are organized as follows:

1. **Basic Integration Tests** (`test_bedrock_integration.py`)
   - Tests initialization and configuration of the Bedrock provider
   - Tests message format conversion
   - Tests API request preparation
   - Tests error handling

2. **Connection Tests** (`test_bedrock_connection_integration.py`)
   - Tests authentication mechanisms
   - Tests connection to Bedrock
   - Tests loading configuration from files

3. **Agent Workflow Tests** (`test_bedrock_agents.py`)
   - Tests Bedrock with agent workflows
   - Tests structured output
   - Tests chain and parallel workflows

## Running the Tests

To run all the Bedrock integration tests:

```bash
pytest tests/integration/bedrock -v
```

To run a specific test file:

```bash
pytest tests/integration/bedrock/test_bedrock_integration.py -v
```

## Using Real AWS Credentials

Most tests use mocked boto3 calls to avoid making actual AWS API calls. To run tests with real AWS credentials, set the environment variable:

```bash
export REAL_BEDROCK_TEST=true
```

This will enable the tests that make actual calls to Bedrock. You'll need valid AWS credentials configured in your environment.

## Configuration

The tests use the configuration in `fastagent.config.yaml`. This file contains settings for:

- The default Bedrock model
- AWS region configuration
- Authentication methods
- Default parameters for different model families

## Adding New Tests

When adding new Bedrock-related integration tests, follow these guidelines:

1. Use the available fixtures in `conftest.py`
2. Mock boto3 calls when possible to avoid actual API calls
3. Test specific features of the Bedrock provider
4. Add appropriate markers (`@pytest.mark.integration`, etc.)
5. Document any new test categories or fixtures

## Test Dependencies

The integration tests depend on:

- boto3 (mocked in most cases)
- pytest and pytest-asyncio
- The fast-agent-mcp package with optional bedrock dependencies