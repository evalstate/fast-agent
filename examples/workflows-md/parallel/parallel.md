---
type: parallel
name: parallel
fan_out:
- proofreader
- fact_checker
- style_enforcer
fan_in: grader
---

    You are a parallel processor that executes multiple agents simultaneously
    and aggregates their results.
