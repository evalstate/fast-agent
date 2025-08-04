# Dynamic Agents Examples

This directory contains examples demonstrating the dynamic agent creation capability in FastAgent. Dynamic agents can be created at runtime based on task analysis, allowing for adaptive team composition.

## Features

- **Runtime Agent Creation**: Create specialized agents on-the-fly
- **Parallel Execution**: Multiple agents can work simultaneously  
- **Lifecycle Management**: Create, use, and terminate agents as needed
- **Tool Access**: Dynamic agents can use MCP servers and tools
- **Tree Display**: Visual representation of agent hierarchy

## Available Examples

### 1. Project Manager (`project_manager.py`)
Demonstrates a project manager that creates and coordinates development teams for software projects.

```bash
# Run the full demo
python project_manager.py

# Interactive mode
python project_manager.py interactive
```

### 2. Simple Demo (`simple_demo.py`)
Basic demonstration of dynamic agent concepts with easy-to-understand examples.

```bash
# Run all simple examples
python simple_demo.py

# Run just the basic example
python simple_demo.py basic

# Run just the delegation example
python simple_demo.py delegation
```

### 3. Code Review Demo (`code_review_demo.py`)
Shows specialized code review teams that analyze code from different perspectives.

```bash
# Run all review examples
python code_review_demo.py

# Run just security-focused review
python code_review_demo.py security

# Run comprehensive review
python code_review_demo.py comprehensive
```

### 4. Interactive Demo (`interactive_demo.py`)
Interactive playground for experimenting with dynamic agents.

```bash
# Interactive mode
python interactive_demo.py

# Guided scenarios
python interactive_demo.py guided

# Quick demonstration
python interactive_demo.py quick
```

### 5. Original Example (`example.py`)
The original comprehensive example with multiple scenarios in one file.

```bash
# Full demo
python example.py

# Simple example
python example.py simple

# Interactive mode
python example.py interactive
```

## How It Works

### 1. Enable Dynamic Agents
```python
@fast.agent(
    name="project_manager",
    dynamic_agents=True,        # Enable dynamic agent creation
    max_dynamic_agents=5,       # Limit to 5 agents
    servers=["filesystem", "fetch"]  # MCP servers available to dynamic agents
)
```

### 2. Create Dynamic Agents
The agent uses tools to create specialists:
```python
# Creates a frontend developer agent
dynamic_agent_create({
    "name": "frontend_dev",
    "instruction": "You are a React/TypeScript expert...",
    "servers": ["filesystem"],
    "tools": {"filesystem": ["read*", "write*"]}
})
```

### 3. Delegate Tasks
```python
# Send task to specific agent
dynamic_agent_send({
    "agent_id": "frontend_dev_abc123", 
    "message": "Create the main App component"
})

# Broadcast to multiple agents (parallel execution)
dynamic_agent_broadcast({
    "message": "Review this code for issues",
    "agent_ids": ["security_expert", "performance_expert"],
    "parallel": true
})
```

## Available Tools

When `dynamic_agents=True`, the agent gets these tools:

- **dynamic_agent_create**: Create new specialized agents
- **dynamic_agent_send**: Send messages to specific agents
- **dynamic_agent_broadcast**: Send messages to multiple agents in parallel
- **dynamic_agent_list**: List all active dynamic agents
- **dynamic_agent_terminate**: Clean up agents when done

## Use Cases

### 1. Development Teams
- Frontend/Backend/Database specialists
- Code reviewers with different focuses
- DevOps and QA specialists

### 2. Content Creation
- Writers, editors, fact-checkers
- Specialized content for different audiences

### 3. Data Analysis
- Data collectors, cleaners, analyzers
- Visualization and reporting specialists

### 4. Research Projects
- Domain experts for different topics
- Fact-checkers and synthesizers

## Architecture

Dynamic agents follow the same patterns as parallel agents:
- **Same Process**: All run in the same Python process
- **Shared Context**: Use the same MCP connections  
- **Separate LLM Contexts**: Each has its own conversation history
- **Parallel Execution**: Use `asyncio.gather()` like ParallelAgent
- **Tree Display**: Extend parallel agent display patterns

## Configuration

Dynamic agents can only use MCP servers defined in `fastagent.config.yaml`. They cannot create new MCP connections, but can be configured with:

- **Different instruction/role**
- **Subset of MCP servers**
- **Filtered tools from those servers**
- **Different models**
- **Own conversation context**

## Limitations

- Maximum number of agents enforced
- Can only use pre-configured MCP servers
- Exist only during parent agent's lifetime
- No persistence across sessions