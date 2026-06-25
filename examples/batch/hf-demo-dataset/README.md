---
title: fast-agent batch demo
pretty_name: fast-agent batch demo
---

# fast-agent batch demo

Small JSONL and text artifacts used by the fast-agent batch processing guide.

Run:

```bash
uvx fast-agent-mcp@latest batch run \
  --input hf://datasets/evalstate/fast-agent-batch-demo/hf-research-questions.jsonl \
  --output hf-research-results.jsonl \
  --agent-card hf://datasets/evalstate/fast-agent-batch-demo/hf-research-agent.md \
  --template hf://datasets/evalstate/fast-agent-batch-demo/hf-research-template.md \
  --limit 3 \
  --id-field id \
  --model kimi26instant
```
