#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""
Extract PR review comments and file evolution from a GitHub Pull Request.

Outputs structured data for LLM analysis of writing style improvements.

Usage:
    uv run extract_pr_reviews.py <pr_url>
    uv run extract_pr_reviews.py <pr_url> --diff      # Show first→final for LLM comparison
    uv run extract_pr_reviews.py <pr_url> --json      # Raw JSON output
    
Examples:
    uv run extract_pr_reviews.py https://github.com/huggingface/blog/pull/3029
    uv run extract_pr_reviews.py huggingface/blog 3029 --diff
"""

import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Optional
from urllib.parse import quote, urlparse


@dataclass
class ReviewComment:
    """A single review comment or suggestion."""
    id: int
    reviewer: str
    path: str
    original_line: Optional[int]
    original_text: str
    comment_type: str  # "suggestion", "feedback", or "reply"
    suggestion_text: Optional[str]
    comment_text: str
    commit_id: str
    created_at: str
    html_url: str
    in_reply_to_id: Optional[int] = None


@dataclass
class FileEvolution:
    """Track a file's content from first to final version."""
    final_path: str
    all_paths: list[str]  # All names this file had during the PR
    first_content: Optional[str] = None
    first_commit: Optional[str] = None
    final_content: Optional[str] = None
    final_commit: Optional[str] = None


@dataclass
class PRReviewData:
    """Complete review data for a PR."""
    owner: str
    repo: str
    pr_number: int
    title: str
    state: str
    first_commit_sha: str
    head_sha: str
    files: list[dict]
    comments: list[ReviewComment]
    commit_history: list[dict]
    file_evolutions: dict = field(default_factory=dict)


def run_gh(args: list[str], check: bool = True) -> dict | list | str:
    """Run gh CLI command and return parsed JSON or raw output."""
    cmd = ["gh"] + args
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        if check:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        return ""
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return result.stdout


def parse_pr_url(url_or_args: list[str]) -> tuple[str, str, int]:
    """Parse PR URL or owner/repo + number into components."""
    if len(url_or_args) == 1:
        url = url_or_args[0]
        parsed = urlparse(url)
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 4 and parts[2] in ("pull", "pulls"):
            return parts[0], parts[1], int(parts[3])
        raise ValueError(f"Invalid PR URL: {url}")
    elif len(url_or_args) == 2:
        owner_repo, pr_num = url_or_args
        if "/" in owner_repo:
            owner, repo = owner_repo.split("/", 1)
            return owner, repo, int(pr_num)
        raise ValueError(f"Expected owner/repo format: {owner_repo}")
    else:
        raise ValueError("Expected PR URL or 'owner/repo pr_number'")


def extract_original_from_diff_hunk(diff_hunk: str, num_lines: int = 1) -> str:
    """Extract the original text that a comment targets from a diff hunk."""
    if not diff_hunk:
        return ""
    lines = diff_hunk.split("\n")
    added_lines = []
    for line in lines:
        if line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:])
    if added_lines:
        return "\n".join(added_lines[-num_lines:]) if num_lines > 1 else added_lines[-1]
    return ""


def parse_suggestion(body: str) -> Optional[str]:
    """Parse suggestion text from a ```suggestion code block."""
    pattern = r"```suggestion\s*\n(.*?)```"
    match = re.search(pattern, body, re.DOTALL)
    if match:
        return match.group(1).rstrip("\n")
    return None


def count_suggestion_lines(body: str) -> int:
    """Count how many lines a suggestion replaces."""
    suggestion = parse_suggestion(body)
    if suggestion:
        return len(suggestion.split("\n"))
    return 1


def get_file_content_at_ref(owner: str, repo: str, path: str, ref: str) -> Optional[str]:
    """Get file content at a specific ref using the contents API."""
    encoded_path = quote(path, safe='')
    
    result = subprocess.run(
        ["gh", "api", f"repos/{owner}/{repo}/contents/{encoded_path}?ref={ref}",
         "-H", "Accept: application/vnd.github.raw"],
        capture_output=True, text=True, check=False
    )
    
    if result.returncode == 0 and result.stdout:
        return result.stdout
    return None


def trace_file_through_commits(owner: str, repo: str, commits: list[dict], final_path: str) -> list[str]:
    """
    Trace a file backwards through commits to find all names it had.
    Returns list of paths from oldest name to newest.
    """
    paths = [final_path]
    current_path = final_path
    
    # Check each commit in reverse order for renames
    for commit in reversed(commits):
        sha = commit["sha"]
        # Get files changed in this commit
        files = run_gh([
            "api", f"repos/{owner}/{repo}/commits/{sha}",
            "--jq", "[.files[] | {filename, previous_filename, status}]"
        ], check=False)
        
        if not files or not isinstance(files, list):
            continue
            
        for f in files:
            if f.get("filename") == current_path and f.get("previous_filename"):
                prev = f["previous_filename"]
                if prev not in paths:
                    paths.insert(0, prev)
                current_path = prev
                break
    
    return paths


def find_first_content(owner: str, repo: str, commits: list[dict], paths: list[str]) -> tuple[Optional[str], Optional[str]]:
    """Find the first version of a file, trying all known paths at each commit."""
    if not commits:
        return None, None
    
    # Try each commit from first onwards until we find the file
    for commit in commits:
        sha = commit["sha"]
        for path in paths:
            content = get_file_content_at_ref(owner, repo, path, sha)
            if content:
                return content, sha[:7]
    
    return None, None


def fetch_pr_data(owner: str, repo: str, pr_number: int, track_evolution: bool = False) -> PRReviewData:
    """Fetch all review data for a PR."""
    repo_ref = f"{owner}/{repo}"
    
    # Get PR metadata
    pr_info = run_gh([
        "pr", "view", str(pr_number),
        "--repo", repo_ref,
        "--json", "title,state,headRefOid,files"
    ])
    
    # Get commits in order
    commits = run_gh([
        "api", f"repos/{repo_ref}/pulls/{pr_number}/commits",
        "--jq", "[.[] | {sha: .sha, message: .commit.message, date: .commit.author.date}]"
    ])
    
    # Get all review comments
    raw_comments = run_gh([
        "api", f"repos/{repo_ref}/pulls/{pr_number}/comments"
    ])
    
    # Process comments
    comments = []
    for c in raw_comments:
        body = c.get("body", "")
        suggestion = parse_suggestion(body)
        num_lines = count_suggestion_lines(body) if suggestion else 1
        original_text = extract_original_from_diff_hunk(c.get("diff_hunk", ""), num_lines)
        
        if suggestion:
            comment_type = "suggestion"
        elif c.get("in_reply_to_id"):
            comment_type = "reply"
        else:
            comment_type = "feedback"
        
        comment = ReviewComment(
            id=c["id"],
            reviewer=c["user"]["login"],
            path=c["path"],
            original_line=c.get("original_line"),
            original_text=original_text,
            comment_type=comment_type,
            suggestion_text=suggestion,
            comment_text=body,
            commit_id=c["commit_id"][:7],
            created_at=c["created_at"],
            html_url=c["html_url"],
            in_reply_to_id=c.get("in_reply_to_id")
        )
        comments.append(comment)
    
    comments.sort(key=lambda x: x.created_at)
    
    # Get files changed (final state)
    files = run_gh([
        "api", f"repos/{repo_ref}/pulls/{pr_number}/files",
        "--jq", "[.[] | {filename: .filename, previous_filename: .previous_filename, status: .status}]"
    ])
    
    first_commit_sha = commits[0]["sha"] if commits else ""
    head_sha = pr_info.get("headRefOid", commits[-1]["sha"] if commits else "")
    
    file_evolutions = {}
    
    if track_evolution:
        # Track text files through their evolution
        for f in files:
            final_path = f["filename"]
            
            # Only track text files
            if not any(final_path.endswith(ext) for ext in ('.md', '.txt', '.rst', '.mdx')):
                continue
            
            # Trace file through commits to find all historical names
            all_paths = trace_file_through_commits(owner, repo, commits, final_path)
            
            # Also add previous_filename from PR files endpoint if present
            if f.get("previous_filename") and f["previous_filename"] not in all_paths:
                all_paths.insert(0, f["previous_filename"])
            
            # Add paths from comments that reference this file
            for c in comments:
                if c.path not in all_paths:
                    # Check if comment path shares basename with any known path
                    c_base = c.path.split("/")[-1]
                    for p in all_paths:
                        if c_base == p.split("/")[-1]:
                            all_paths.append(c.path)
                            break
            
            # Dedupe while preserving order
            all_paths = list(dict.fromkeys(all_paths))
            
            evo = FileEvolution(final_path=final_path, all_paths=all_paths)
            
            # Get first version (try all paths)
            evo.first_content, evo.first_commit = find_first_content(owner, repo, commits, all_paths)
            
            # Get final version
            evo.final_content = get_file_content_at_ref(owner, repo, final_path, head_sha)
            evo.final_commit = head_sha[:7]
            
            file_evolutions[final_path] = evo
    
    return PRReviewData(
        owner=owner,
        repo=repo,
        pr_number=pr_number,
        title=pr_info["title"],
        state=pr_info["state"],
        first_commit_sha=first_commit_sha[:7],
        head_sha=head_sha[:7],
        files=files,
        comments=comments,
        commit_history=commits,
        file_evolutions=file_evolutions
    )


def format_suggestions_and_feedback(data: PRReviewData) -> str:
    """Format just the suggestions and feedback (no file content)."""
    lines = []
    lines.append(f"# PR Review Analysis: {data.title}")
    lines.append(f"**PR:** {data.owner}/{data.repo}#{data.pr_number}")
    lines.append(f"**State:** {data.state}")
    lines.append(f"**Commits:** {len(data.commit_history)} total")
    lines.append("")
    
    # Files
    lines.append("## Files Changed")
    for f in data.files:
        prev = f.get("previous_filename")
        if prev:
            lines.append(f"- {f['filename']} ← *renamed from {prev}*")
        else:
            lines.append(f"- {f['filename']} ({f['status']})")
    lines.append("")
    
    # Suggestions
    suggestions = [c for c in data.comments if c.comment_type == "suggestion"]
    if suggestions:
        lines.append(f"## Writing Suggestions ({len(suggestions)})")
        lines.append("")
        
        reviewers = {}
        for s in suggestions:
            reviewers.setdefault(s.reviewer, []).append(s)
        
        for reviewer, items in reviewers.items():
            lines.append(f"### @{reviewer} ({len(items)} suggestions)")
            lines.append("")
            for i, s in enumerate(items, 1):
                lines.append(f"**{i}. Line {s.original_line or '?'}** (`{s.path}`)")
                lines.append("")
                lines.append("Original:")
                for line in s.original_text.split("\n"):
                    lines.append(f"> {line}")
                lines.append("")
                lines.append("Suggested:")
                for line in (s.suggestion_text or "").split("\n"):
                    lines.append(f"> {line}")
                lines.append("")
    
    # Feedback
    feedback = [c for c in data.comments if c.comment_type == "feedback"]
    if feedback:
        lines.append(f"## Reviewer Feedback ({len(feedback)})")
        lines.append("")
        for i, c in enumerate(feedback, 1):
            lines.append(f"### {i}. @{c.reviewer} on `{c.path}` line {c.original_line or '?'}")
            lines.append("")
            lines.append(c.comment_text)
            lines.append("")
            lines.append(f"[View on GitHub]({c.html_url})")
            lines.append("")
    
    return "\n".join(lines)


def format_diff_comparison(data: PRReviewData) -> str:
    """Format for LLM paragraph-by-paragraph comparison."""
    lines = []
    lines.append(f"# PR Style Analysis: {data.title}")
    lines.append(f"**PR:** {data.owner}/{data.repo}#{data.pr_number}")
    lines.append("")
    
    # First: the explicit suggestions (these are precise before/after)
    suggestions = [c for c in data.comments if c.comment_type == "suggestion"]
    if suggestions:
        lines.append("## Explicit Suggestions (exact before → after)")
        lines.append("")
        for i, s in enumerate(suggestions, 1):
            lines.append(f"### {i}. @{s.reviewer}")
            lines.append(f"**Before:** {s.original_text}")
            lines.append(f"**After:** {s.suggestion_text}")
            lines.append("")
    
    # Second: the feedback comments (context for what reviewers asked for)
    feedback = [c for c in data.comments if c.comment_type == "feedback"]
    if feedback:
        lines.append("## Reviewer Feedback (requests without explicit replacement)")
        lines.append("")
        for i, c in enumerate(feedback, 1):
            lines.append(f"{i}. **@{c.reviewer}**: {c.comment_text}")
            lines.append("")
    
    # Third: full file comparison for each text file
    if data.file_evolutions:
        lines.append("---")
        lines.append("")
        lines.append("## File Evolution (first draft → final)")
        lines.append("")
        lines.append("*Compare paragraph-by-paragraph to see how the author responded to feedback.*")
        lines.append("")
        
        for path, evo in data.file_evolutions.items():
            if not evo.first_content and not evo.final_content:
                continue
            
            lines.append(f"### {path}")
            if len(evo.all_paths) > 1:
                lines.append(f"*File was renamed: {" → ".join(evo.all_paths)}*")
            lines.append("")
            
            if evo.first_content:
                lines.append(f"#### FIRST DRAFT ({evo.first_commit})")
                lines.append("```markdown")
                lines.append(evo.first_content)
                lines.append("```")
                lines.append("")
            
            if evo.final_content:
                lines.append(f"#### FINAL VERSION ({evo.final_commit})")
                lines.append("```markdown")
                lines.append(evo.final_content)
                lines.append("```")
                lines.append("")
    
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    output_json = "--json" in sys.argv
    show_diff = "--diff" in sys.argv
    
    try:
        owner, repo, pr_number = parse_pr_url(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # --diff implies we need evolution tracking
    track_evolution = show_diff
    
    try:
        data = fetch_pr_data(owner, repo, pr_number, track_evolution=track_evolution)
    except subprocess.CalledProcessError as e:
        print(f"Error fetching PR data: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    
    if output_json:
        output = {
            "owner": data.owner,
            "repo": data.repo,
            "pr_number": data.pr_number,
            "title": data.title,
            "state": data.state,
            "first_commit_sha": data.first_commit_sha,
            "head_sha": data.head_sha,
            "files": data.files,
            "commit_history": data.commit_history,
            "comments": [asdict(c) for c in data.comments],
        }
        if track_evolution:
            output["file_evolutions"] = {
                path: {
                    "all_paths": evo.all_paths,
                    "first_content": evo.first_content,
                    "first_commit": evo.first_commit,
                    "final_content": evo.final_content,
                    "final_commit": evo.final_commit,
                }
                for path, evo in data.file_evolutions.items()
            }
        print(json.dumps(output, indent=2))
    elif show_diff:
        print(format_diff_comparison(data))
    else:
        print(format_suggestions_and_feedback(data))


if __name__ == "__main__":
    main()
