# Session Support Integration Guide

## What was implemented:

1. **Session Manager** (`src/fast_agent/session/session_manager.py`):
   - `SessionManager`: Main class managing sessions in `.fast-agent/sessions/`
   - `Session`: Represents a single session with history files
   - `SessionInfo`: Metadata about a session

2. **Session Commands** (`src/fast_agent/ui/session_commands.py`):
   - `handle_list_sessions_cmd`: List all sessions
   - `handle_create_session_cmd`: Create new session
   - `handle_switch_session_cmd`: Switch to existing session
   - `handle_save_history_cmd`: Save history to current session
   - `handle_session_command`: Parse and dispatch session commands

3. **Updated Command Payloads** (`src/fast_agent/ui/command_payloads.py`):
   - Added `ListSessionsCommand`
   - Added `CreateSessionCommand`
   - Added `SwitchSessionCommand`

## Integration Steps:

### 1. Update enhanced_prompt.py

Add these imports at the top:
```python
from fast_agent.ui.session_commands import (
    handle_session_command,
    handle_save_history_cmd,
)
```

Update the HELP text in `handle_special_commands()`:
```python
elif command == "HELP":
    # ... existing help text ...
    rich_print("  /sessions      - List all available sessions")
    rich_print("  /create_session [name] - Create a new session")
    rich_print("  /switch_session <name> - Switch to a different session")
    # ... rest of help text ...
```

Add session command handling before the final `return False`:
```python
# Handle session commands
session_result = await handle_session_command(command, agent_app)
if session_result is not False:
    return session_result
```

Update the SAVE_HISTORY handler to be session-aware:
```python
elif isinstance(command, str) and command.startswith("SAVE_HISTORY"):
    # Parse filename from command
    parts = command.split(maxsplit=1)
    filename = parts[1] if len(parts) > 1 else None
    
    cmd = SaveHistoryCommand(filename=filename)
    return await handle_save_history_cmd(cmd, agent_app)
```

### 2. Update AgentApp to support sessions

Add session integration to `src/fast_agent/core/agent_app.py`:

```python
from fast_agent.session import SessionManager

class AgentApp:
    def __init__(self, ...):
        # ... existing initialization ...
        self._session_manager = SessionManager()
        
    @property
    def session_manager(self) -> SessionManager:
        """Get the session manager."""
        return self._session_manager
        
    async def save_current_session(self) -> str | None:
        """Save current agent history to session."""
        if not self._agents:
            return None
            
        # Use default agent or first available agent
        agent_name = self._default_agent_name
        if agent_name not in self._agents and self._agents:
            agent_name = next(iter(self._agents.keys()))
            
        agent = self._agents.get(agent_name)
        if not agent:
            return None
            
        return self._session_manager.save_current_session(agent)
```

### 3. Optional: Auto-save on exit

Add auto-save functionality by updating the interactive loop in `agent_app.py`:

```python
async def interactive(...):
    # ... existing code ...
    try:
        while True:
            # ... existing loop code ...
            pass
    finally:
        # Auto-save to session on exit
        if self._session_manager.current_session:
            try:
                agent_name = self._default_agent_name or next(iter(self._agents.keys()))
                agent = self._agents.get(agent_name)
                if agent:
                    self._session_manager.save_current_session(agent)
                    logger.info(f"Auto-saved session: {self._session_manager.current_session.info.name}")
            except Exception as e:
                logger.error(f"Failed to auto-save session: {e}")
```

## Usage Examples:

### List sessions
```
/sessions
```

### Create a new session
```
/create_session my_project
```

### Switch to a session
```
/switch_session my_project
```

### Save history (automatically goes to current session)
```
/save_history
```

### Save with custom filename
```
/save_history brainstorming_results.json
```

## Session Directory Structure:

```
.fast-agent/
└── sessions/
    ├── session_240115_143022/
    │   ├── session.json
    │   └── history_240115_143022.json
    └── my_project/
        ├── session.json
        ├── history_240115_143100.json
        └── history_240115_150000.json
```

Each session directory contains:
- `session.json`: Session metadata
- `history_*.json`: Conversation history files

## Benefits:

1. **Organized History**: Group conversations by project/topic
2. **Automatic Saving**: `/save_history` automatically saves to current session
3. **Easy Switching**: Switch between contexts without losing history
4. **Clean File Management**: All session data in `.fast-agent/sessions/`
5. **Metadata Tracking**: Creation date, last activity, history file list
