# ACP Tool Calls Implementation Assessment

**Date:** 2025-11-11
**Assessor:** Claude
**Specification:** https://agentclientprotocol.com/llms.txt (Tool Calls section)

## Executive Summary

Three implementations of the ACP Tool Calls specification were reviewed. **Branch 2 (claude/acp-tool-calls-implementation-011CV18RQDfYLTy2oRptywSJ)** is recommended as the most appropriate implementation due to its clean architecture, no core code modifications, good documentation, and maintainable design.

## Implementations Reviewed

### Branch 1: claude/acp-tool-calls-implementation-011CV18UGQwtc2PbWpSCBcHf
- **Size:** 2,193 lines across 8 files
- **Core Changes:** None
- **Tests:** 25 unit tests, placeholder integration tests
- **Documentation:** None

### Branch 2: claude/acp-tool-calls-implementation-011CV18RQDfYLTy2oRptywSJ ⭐ **RECOMMENDED**
- **Size:** 1,828 lines across 9 files
- **Core Changes:** None
- **Tests:** 9 unit tests, placeholder integration tests
- **Documentation:** ACP_TOOL_CALLS.md

### Branch 3: claude/agent-client-protocol-docs-011CV18QMgaj4yWvhUGbCp5m
- **Size:** 1,141 lines across 7 files
- **Core Changes:** Modifies mcp_aggregator.py (~100 lines)
- **Tests:** Minimal unit tests, functional integration tests
- **Documentation:** ACP_TOOL_CALLS.md

## Evaluation Criteria

### 1. Clean Impact on Codebase

| Branch | Core Modifications | Compartmentalization | Score |
|--------|-------------------|---------------------|-------|
| Branch 1 | ✅ None | ✅ Excellent | 10/10 |
| Branch 2 | ✅ None | ✅ Excellent | 10/10 |
| Branch 3 | ❌ mcp_aggregator.py | ⚠️ Mixed | 5/10 |

**Winner:** Branch 1 & 2 (tie)

Branch 3's modifications to `mcp_aggregator.py` introduce ACP-specific hooks into core MCP infrastructure, creating coupling and maintenance burden.

### 2. Integration Tests

| Branch | Test Quality | Functionality | Coverage |
|--------|-------------|--------------|----------|
| Branch 1 | Placeholder | Non-functional | 3/10 |
| Branch 2 | Placeholder | Non-functional | 3/10 |
| Branch 3 | Functional | Uses ***CALL_TOOL | 9/10 |

**Winner:** Branch 3

Branch 3's integration tests actually exercise the tool call lifecycle using the passthrough model's `***CALL_TOOL` directive, validating real behavior.

### 3. Code Quality & Maintainability

| Branch | Architecture | Clarity | Documentation |
|--------|-------------|---------|---------------|
| Branch 1 | Factory + Wrapper | Good | None |
| Branch 2 | Middleware + Wrapper | Excellent | Comprehensive |
| Branch 3 | Hook-based | Good | Comprehensive |

**Winner:** Branch 2

Branch 2 uses dataclasses and a clear middleware pattern that's easy to understand and maintain. The documentation is thorough and helpful.

### 4. Implementation Size

| Branch | Lines | Files | Complexity |
|--------|-------|-------|-----------|
| Branch 1 | 2,193 | 8 | High |
| Branch 2 | 1,828 | 9 | Medium |
| Branch 3 | 1,141 | 7 | Low |

**Winner:** Branch 3

Smaller implementations are generally easier to maintain, though this must be balanced against other factors.

## Detailed Analysis

### Branch 1: Comprehensive but Complex

**Architecture:**
```
ToolCallManager (350 lines)
  ├── Lifecycle tracking
  ├── Tool kind inference
  └── ACP notifications

ToolPermissionManager (314 lines)
  ├── Permission caching
  ├── Decorator support
  └── Factory pattern

ACPToolWrapper (231 lines)
  ├── Wraps MCPAggregator
  ├── Permission checks
  └── Progress forwarding
```

**Strengths:**
- Most comprehensive unit tests (25 tests)
- Sophisticated permission caching
- Follows elicitation handler pattern
- No core code changes

**Weaknesses:**
- Largest implementation
- No documentation
- Integration tests don't validate functionality
- Complex multi-layer architecture

### Branch 2: Clean and Balanced ⭐

**Architecture:**
```
ToolCallTracker (185 lines, dataclass-based)
  ├── Simple state management
  └── ACP notifications

ACPToolCallMiddleware (264 lines)
  ├── Tool call wrapping
  ├── Permission checking
  ├── Progress forwarding
  └── Kind inference

ACPAgentWrapper (71 lines)
  └── Transparent agent wrapping
```

**Strengths:**
- Clean, modular architecture
- Good documentation (ACP_TOOL_CALLS.md)
- No core code changes
- Dataclass-based for clarity
- Middleware pattern is intuitive

**Weaknesses:**
- Fewer unit tests than Branch 1
- Integration tests are placeholders

**Why This is Recommended:**
1. **No core modifications** - Completely compartmentalized to `acp/` package
2. **Clear architecture** - Middleware pattern is well-understood and maintainable
3. **Good documentation** - Explains architecture, usage, and integration
4. **Right-sized** - Not too complex, not too simple
5. **Extensible** - Easy to add features or adapt tests from Branch 3

### Branch 3: Best Tests, Invasive Changes

**Architecture:**
```
ACPToolProgressManager (354 lines)
  ├── Hook-based integration
  └── Tool call tracking

Modified MCPAggregator (+~100 lines)
  ├── tool_start_hook parameter
  ├── tool_progress_hook parameter
  └── tool_complete_hook parameter
```

**Strengths:**
- Best integration tests (functional)
- Smallest implementation
- Clean hook design
- Good documentation

**Weaknesses:**
- Modifies core agent code
- Creates coupling between ACP and MCP layers
- Increases MCPAggregator API surface
- May complicate future refactoring

**Core Code Impact:**
The modifications to `mcp_aggregator.py` add three optional hook parameters to the constructor and sprinkle hook calls throughout `call_tool()`:

```python
# New in __init__:
self._tool_start_hook = tool_start_hook
self._tool_progress_hook = tool_progress_hook
self._tool_complete_hook = tool_complete_hook

# In call_tool():
if self._tool_start_hook:
    tool_call_id = await self._tool_start_hook(...)

if self._tool_progress_hook and tool_call_id:
    await self._tool_progress_hook(...)

if self._tool_complete_hook and tool_call_id:
    await self._tool_complete_hook(...)
```

While these hooks are optional and backward-compatible, they introduce ACP-specific concerns into the core MCP layer.

## Recommendation: Branch 2

**Selected Implementation:** `claude/acp-tool-calls-implementation-011CV18RQDfYLTy2oRptywSJ`

### Rationale

Branch 2 provides the best balance of:
1. ✅ **Clean codebase impact** - No core modifications
2. ✅ **Maintainable architecture** - Clear middleware pattern
3. ✅ **Good documentation** - Comprehensive guide included
4. ✅ **Adequate testing** - Can be enhanced with Branch 3's test approach

While Branch 3 has superior integration tests, its modifications to core agent code create a maintenance burden that outweighs the testing benefits. Branch 1 is solid but unnecessarily complex.

### Enhancement Recommendation

To get the best of both worlds:
1. Use Branch 2 as the base implementation
2. Adapt Branch 3's integration tests (the `***CALL_TOOL` approach)
3. Add 5-10 more unit tests for edge cases
4. Result: Clean architecture + comprehensive tests

### Implementation Notes

**What makes Branch 2 "clean":**
- All ACP tool call code lives in `src/fast_agent/acp/`
- No changes to `mcp_aggregator.py` or other core files
- Uses wrapper pattern to intercept calls transparently
- Can be removed entirely without affecting core functionality

**Integration approach:**
```python
# In agent_acp_server.py, during session creation:
tracker = ToolCallTracker(session_id, connection)
middleware = ACPToolCallMiddleware(tracker, permission_handler)
wrapped_agent = ACPAgentWrapper(agent, middleware)
```

This approach keeps ACP concerns isolated and makes testing straightforward.

## Conclusion

**Recommended:** Branch 2 (`claude/acp-tool-calls-implementation-011CV18RQDfYLTy2oRptywSJ`)

**Next Steps:**
1. Merge Branch 2 implementation
2. Enhance with functional integration tests from Branch 3
3. Add edge case unit tests
4. Consider borrowing permission caching details from Branch 1 if needed

The combination of clean architecture, no core code changes, and good documentation makes Branch 2 the most maintainable long-term choice for the fast-agent codebase.
