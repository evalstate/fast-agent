---
name: pr-writing-review
description: Extract and analyze writing improvements from GitHub PR review comments. Use when asked to show review feedback, style changes, or editorial improvements from a GitHub pull request URL. Handles both explicit suggestions and plain text feedback. Produces structured output comparing original phrasing with reviewer suggestions to help refine future writing.
---

# PR Writing Review

Extract editorial feedback from GitHub PRs to learn from review improvements.

## Division of Labor

| Tool | Responsibility |
|------|----------------|
| **Python script** | API calls, parsing, file tracking across renames, structured extraction |
| **LLM analysis** | Pattern recognition, paragraph comparison, style lesson synthesis |

## Quick Start

```bash
# Get suggestions and feedback
uv run scripts/extract_pr_reviews.py <pr_url>

# Get full first→final comparison for deep analysis
uv run scripts/extract_pr_reviews.py <pr_url> --diff
```

## Workflow

### Step 1: Extract with `--diff`

```bash
uv run scripts/extract_pr_reviews.py https://github.com/org/repo/pull/123 --diff
```

This outputs:
1. **Explicit Suggestions** — exact before/after text from `suggestion` blocks
2. **Reviewer Feedback** — plain text comments (the "why" behind changes)
3. **File Evolution** — first draft and final version of each text file

### Step 2: Analyze the Output

With the script output, perform this analysis:

#### A. Catalog the Explicit Suggestions

Create a table of mechanical fixes:

| Pattern | Original | Fixed |
|---------|----------|-------|
| Grammar | "Its easier" | "It's easier" |
| Filler removal | "using this way" | "this way" |
| Capitalization | "Image Generation" | "image generation" |

#### B. Map Feedback to Changes

For each reviewer feedback comment:
1. Find the relevant section in FIRST DRAFT
2. Find the same section in FINAL VERSION  
3. Document what changed and why

Example:
> **Feedback:** "would be nice to end more enthusiastically"  
> **First draft:** "...it's simple to add new tools to Claude and use them straight away."  
> **Final:** "...Let us know what you find and create in the comments below!"  
> **Lesson:** End blog posts with a call-to-action

#### C. Paragraph-by-Paragraph Comparison

Compare FIRST DRAFT to FINAL VERSION section by section:
- What was added?
- What was removed?
- What was reworded?
- What structural changes were made?

#### D. Synthesize Style Patterns

Group findings into categories:

| Category | Patterns Found |
|----------|----------------|
| **Clarity** | Passive→active, shorter sentences, remove filler |
| **Precision** | Vague→specific, "Create"→"Generate" |
| **Tone** | Added enthusiasm, call-to-action endings |
| **Structure** | Added transitions, better section flow |
| **Grammar** | its/it's, subject-verb agreement |
| **Content** | Added links, examples, context |

## Script Options

> 📁 **All paths are relative to the directory containing this SKILL.md
file.**
> Before running any script, first `cd` to that directory or use the full
path.

| Flag | Output | Use Case |
|------|--------|----------|
| *(none)* | Suggestions + feedback | Quick review of what reviewers said |
| `--diff` | Above + full file versions | Deep analysis of how author responded |
| `--json` | Raw JSON | Programmatic processing |

## Output Structure

### Default Output
- **Writing Suggestions**: Grouped by reviewer, shows original→suggested text
- **Reviewer Feedback**: Plain comments without code suggestions

### With `--diff`
- **Explicit Suggestions**: Compact before/after pairs
- **Reviewer Feedback**: Numbered list of requests
- **File Evolution**: Full FIRST DRAFT and FINAL VERSION for each .md/.txt/.rst file

## Handling File Renames

The script automatically traces files through renames by:
1. Checking each commit for rename operations
2. Building a path history (e.g., `claudeimages.md` → `claude-images.md` → `claude-and-mcp.md`)
3. Fetching content using the correct path for each commit

## Example Analysis Output

After running the script and performing LLM analysis, produce a summary like:

```markdown
## Style Lessons from PR #123

### Mechanical Fixes
- Fix grammar: "Its" → "It's" (contraction)
- Lowercase generic terms: "Image Generation" → "image generation"
- Remove filler: "the output quality of" → "the quality of"

### Reviewer-Driven Changes  
- **"end more enthusiastically"** → Added call-to-action in conclusion
- **"emphasize these are SoTA"** → Changed "latest" to "state-of-the-art"
- **"add blurb about MCP Server"** → Added explanatory paragraph

### Structural Improvements
- Added transition sentence between sections
- Simplified setup instructions (3 sentences → 1)
- Added new bullet point for model flexibility
```

## Limitations

- Only extracts inline PR review comments (not issue comments or PR description)
- File content retrieval requires files to exist in the git tree
- Very long files may need chunked analysis

{{currentDate}}
{{env}}
