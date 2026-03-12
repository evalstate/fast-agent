---
name: code-review
description: Protocol for reviewing code and providing structured verdicts
---

# Code Review Skill

You are performing a code review. Follow this protocol for consistent, actionable feedback.

## Review Process

### Step 1: Understand the Context
- Read the project brief and any related documentation in the workspace
- Understand what the author was trying to accomplish
- Check `team_roster.json` to know who wrote what

### Step 2: Review the Code
- Read all relevant files in the workspace
- Check for:
  - **Correctness** — Does the code do what it's supposed to?
  - **Quality** — Is it readable, maintainable, well-structured?
  - **Edge cases** — Are there unhandled scenarios?
  - **Security** — Any obvious security issues?
  - **Performance** — Any obvious performance problems?

### Step 3: Write Your Review
Write your review to: `reviews/<step_name>_review.md`

Include:
1. **Summary** — Overall assessment
2. **Issues found** — Specific problems with file/line references
3. **Suggestions** — Improvements that aren't blockers
4. **Verdict** — Your final decision

### Step 4: Declare Verdict

**Always end your review with one of these exact strings:**

✅ When code is acceptable:
```
VERDICT: PASS — <brief reason>
```

❌ When code needs fixes:
```
VERDICT: FAIL — <brief reason with key issues>
```

## Communication

After writing your review:
- Use `send_message_to_agent(to_role="<author_role>", message="...")` to notify the author
- Include a summary of key issues if FAIL
- Be constructive — suggest fixes, don't just point out problems

## As a Code Author (receiving review)

When you receive a FAIL verdict:
1. Read the review feedback carefully
2. Fix the identified issues in your workspace files
3. Message the reviewer: `send_message_to_agent(to_role="<reviewer_role>", message="Fixes applied, ready for re-review")`
