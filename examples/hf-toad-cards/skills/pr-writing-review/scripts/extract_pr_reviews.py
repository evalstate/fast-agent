#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Extract PR review comments and file evolution from a GitHub Pull Request.

Outputs structured data for LLM analysis of writing style improvements.

Usage:
    uv run scripts/extract_pr_reviews.py <pr_url>
    uv run scripts/extract_pr_reviews.py <pr_url> --diff      # Show first→final for LLM comparison
    uv run scripts/extract_pr_reviews.py <pr_url> --json      # Raw JSON output

Examples:
    uv run scripts/extract_pr_reviews.py https://github.com/huggingface/blog/pull/3029
    uv run scripts/extract_pr_reviews.py huggingface/blog 3029 --diff
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Optional
from urllib.parse import quote, urlparse


# -----------------
# Data structures
# -----------------


@dataclass
class ReviewComment:
    """A single review comment or suggestion."""

    id: int
    reviewer: str
    path: str
    original_line: Optional[int]
    original_text: str
    comment_type: str  # "suggestion", "feedback", or "reply"
    suggestion_texts: list[str] = field(default_factory=list)  # supports multiple suggestion blocks
    comment_text: str = ""
    commit_id: str = ""
    created_at: str = ""
    html_url: str = ""
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
    file_evolutions: dict[str, FileEvolution] = field(default_factory=dict)


# -----------------
# gh helpers
# -----------------


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


def run_gh_jsonlines(args: list[str], check: bool = True) -> list[dict]:
    """Run gh command that prints one JSON object per line; return list of objects."""
    cmd = ["gh"] + args
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        if check:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
        return []

    out: list[dict] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


# -----------------
# Parsing helpers
# -----------------


def parse_pr_url(url_or_args: list[str]) -> tuple[str, str, int]:
    """Parse PR URL or owner/repo + number into components."""
    if len(url_or_args) == 1:
        url = url_or_args[0]
        parsed = urlparse(url)
        parts = parsed.path.strip("/").split("/")
        if len(parts) >= 4 and parts[2] in ("pull", "pulls"):
            return parts[0], parts[1], int(parts[3])
        raise ValueError(f"Invalid PR URL: {url}")

    if len(url_or_args) == 2:
        owner_repo, pr_num = url_or_args
        if "/" in owner_repo:
            owner, repo = owner_repo.split("/", 1)
            return owner, repo, int(pr_num)
        raise ValueError(f"Expected owner/repo format: {owner_repo}")

    if len(url_or_args) == 3:
        owner, repo, pr_num = url_or_args
        return owner, repo, int(pr_num)

    raise ValueError("Expected PR URL or 'owner/repo pr_number'")


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def extract_original_from_diff_hunk(diff_hunk: str, num_lines: int = 1) -> str:
    """Extract the original text a comment targets from a diff hunk.

    For PR review comments, GitHub provides the diff hunk. Inline comments are
    typically anchored to the added lines, so we extract the last N added lines.
    """

    if not diff_hunk:
        return ""

    lines = normalize_newlines(diff_hunk).split("\n")
    added_lines: list[str] = []
    for line in lines:
        if line.startswith("+") and not line.startswith("+++"):
            added_lines.append(line[1:])

    if not added_lines:
        return ""

    num_lines = max(1, num_lines)
    if num_lines == 1:
        return added_lines[-1]
    return "\n".join(added_lines[-num_lines:])


_SUGGESTION_BLOCK_RE = re.compile(
    r"```suggestion(?:[^\n`]*)?\s*\n(.*?)```", re.DOTALL
)


def parse_suggestions(body: str) -> list[str]:
    """Extract all ```suggestion blocks from a comment body.

    Supports GitHub variants like:
    - ```suggestion
    - ```suggestion:-0+1
    and returns *all* suggestion blocks found.
    """

    if not body:
        return []

    body = normalize_newlines(body)

    suggestions: list[str] = []
    for m in _SUGGESTION_BLOCK_RE.finditer(body):
        s = m.group(1)
        # Keep internal newlines but strip trailing whitespace/newlines
        s = normalize_newlines(s).rstrip()
        suggestions.append(s)

    return suggestions


def count_suggestion_lines(suggestion_text: str) -> int:
    """Count how many lines a suggestion replaces."""

    if not suggestion_text:
        return 1
    return len(normalize_newlines(suggestion_text).split("\n"))


# -----------------
# Content helpers
# -----------------


def get_file_content_at_ref(owner: str, repo: str, path: str, ref: str) -> Optional[str]:
    """Get file content at a specific ref using the contents API."""

    encoded_path = quote(path, safe="")
    result = subprocess.run(
        [
            "gh",
            "api",
            f"repos/{owner}/{repo}/contents/{encoded_path}?ref={ref}",
            "-H",
            "Accept: application/vnd.github.raw",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode == 0 and result.stdout:
        return result.stdout
    return None


def trace_file_through_commits(owner: str, repo: str, commits: list[dict], final_path: str) -> list[str]:
    """Trace a file backwards through commits to find all names it had.

    Returns a list of paths from oldest name to newest.
    """

    paths = [final_path]
    current_path = final_path

    # Check each commit in reverse order for renames.
    for commit in reversed(commits):
        sha = commit["sha"]
        files = run_gh(
            [
                "api",
                f"repos/{owner}/{repo}/commits/{sha}",
                "--jq",
                "[.files[] | {filename, previous_filename, status}]",
            ],
            check=False,
        )

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


def find_first_content(
    owner: str, repo: str, commits: list[dict], paths: list[str]
) -> tuple[Optional[str], Optional[str]]:
    """Find the first version of a file, trying all known paths at each commit."""

    if not commits:
        return None, None

    for commit in commits:
        sha = commit["sha"]
        for path in paths:
            content = get_file_content_at_ref(owner, repo, path, sha)
            if content:
                return content, sha[:7]

    return None, None


# -----------------
# Main fetch
# -----------------


def fetch_pr_data(owner: str, repo: str, pr_number: int, track_evolution: bool = False) -> PRReviewData:
    """Fetch all review data for a PR."""

    repo_ref = f"{owner}/{repo}"

    pr_info = run_gh(
        [
            "pr",
            "view",
            str(pr_number),
            "--repo",
            repo_ref,
            "--json",
            "title,state,headRefOid,files",
        ]
    )

    # Paginated commits in order (JSON-lines).
    commits = run_gh_jsonlines(
        [
            "api",
            f"repos/{repo_ref}/pulls/{pr_number}/commits",
            "--paginate",
            "--jq",
            ".[] | {sha: .sha, message: .commit.message, date: .commit.author.date}",
        ]
    )

    # Paginated review comments.
    raw_comments = run_gh_jsonlines(
        [
            "api",
            f"repos/{repo_ref}/pulls/{pr_number}/comments",
            "--paginate",
            "--jq",
            ".[]",
        ]
    )

    comments: list[ReviewComment] = []
    for c in raw_comments:
        body = c.get("body", "") or ""
        suggestion_texts = parse_suggestions(body)
        clean_body = normalize_newlines(body).strip()

        if suggestion_texts:
            comment_type = "suggestion"
            # Remove suggestion fences from reviewer note output.
            clean_body = normalize_newlines(_SUGGESTION_BLOCK_RE.sub("", body)).strip()
            # For extracting original text, use the *largest* suggestion block length.
            num_lines = max(count_suggestion_lines(s) for s in suggestion_texts)
        elif c.get("in_reply_to_id"):
            comment_type = "reply"
            num_lines = 1
        else:
            comment_type = "feedback"
            num_lines = 1

        original_text = extract_original_from_diff_hunk(c.get("diff_hunk", "") or "", num_lines)

        comments.append(
            ReviewComment(
                id=c["id"],
                reviewer=c["user"]["login"],
                path=c.get("path") or "",
                original_line=c.get("original_line"),
                original_text=original_text,
                comment_type=comment_type,
                suggestion_texts=suggestion_texts,
                comment_text=clean_body,
                commit_id=(c.get("commit_id") or "")[:7],
                created_at=c.get("created_at") or "",
                html_url=c.get("html_url") or "",
                in_reply_to_id=c.get("in_reply_to_id"),
            )
        )

    comments.sort(key=lambda x: x.created_at)

    # Paginated PR files.
    files = run_gh_jsonlines(
        [
            "api",
            f"repos/{repo_ref}/pulls/{pr_number}/files",
            "--paginate",
            "--jq",
            ".[] | {filename: .filename, previous_filename: .previous_filename, status: .status}",
        ]
    )

    first_commit_sha = commits[0]["sha"] if commits else ""
    head_sha = pr_info.get("headRefOid", commits[-1]["sha"] if commits else "")

    file_evolutions: dict[str, FileEvolution] = {}
    if track_evolution:
        for f in files:
            final_path = f["filename"]

            # Only track text files.
            if not any(final_path.endswith(ext) for ext in (".md", ".txt", ".rst", ".mdx")):
                continue

            all_paths = trace_file_through_commits(owner, repo, commits, final_path)

            # Also include previous_filename from PR files endpoint if present.
            if f.get("previous_filename") and f["previous_filename"] not in all_paths:
                all_paths.insert(0, f["previous_filename"])

            # Add paths from comments that reference this file (best-effort).
            for c in comments:
                if c.path and c.path not in all_paths:
                    c_base = c.path.split("/")[-1]
                    if any(c_base == p.split("/")[-1] for p in all_paths):
                        all_paths.append(c.path)

            # Dedupe while preserving order.
            all_paths = list(dict.fromkeys(all_paths))

            evo = FileEvolution(final_path=final_path, all_paths=all_paths)
            evo.first_content, evo.first_commit = find_first_content(owner, repo, commits, all_paths)
            evo.final_content = get_file_content_at_ref(owner, repo, final_path, head_sha)
            evo.final_commit = head_sha[:7] if head_sha else None

            file_evolutions[final_path] = evo

    return PRReviewData(
        owner=owner,
        repo=repo,
        pr_number=pr_number,
        title=pr_info["title"],
        state=pr_info["state"],
        first_commit_sha=first_commit_sha[:7] if first_commit_sha else "",
        head_sha=head_sha[:7] if head_sha else "",
        files=files,
        comments=comments,
        commit_history=commits,
        file_evolutions=file_evolutions,
    )


# -----------------
# Output formatting
# -----------------


def fenced_block(text: str, lang: str = "text") -> str:
    text = text or ""
    text = normalize_newlines(text).rstrip("\n")
    return f"```{lang}\n{text}\n```"




def maybe_truncate_text(text: str, limit: int | None) -> str:
    if not limit or limit <= 0 or len(text) <= limit:
        return text
    truncated = text[: max(0, limit - 1)].rstrip()
    return f"{truncated}\n\n...[truncated {len(text) - len(truncated)} chars]"


def format_suggestions_and_feedback(data: PRReviewData) -> str:
    """Format just the suggestions and feedback (no file content)."""

    lines: list[str] = []
    lines.append(f"# PR Review Analysis: {data.title}")
    lines.append(f"**PR:** {data.owner}/{data.repo}#{data.pr_number}")
    lines.append(f"**State:** {data.state}")
    lines.append(f"**Commits:** {len(data.commit_history)} total")
    lines.append("")

    # Files
    lines.append("## Files Changed")
    for f in data.files:
        prev = f.get("previous_filename")
        status = f.get("status", "modified")
        if prev:
            lines.append(f"- {f['filename']} ({status}) ← *renamed from {prev}*")
        else:
            lines.append(f"- {f['filename']} ({status})")
    lines.append("")

    # Suggestions
    suggestions = [c for c in data.comments if c.comment_type == "suggestion"]
    if suggestions:
        total_suggestions = sum(len(c.suggestion_texts) for c in suggestions)
        lines.append(f"## Writing Suggestions ({total_suggestions})")
        lines.append("")

        reviewers: dict[str, list[ReviewComment]] = {}
        for s in suggestions:
            reviewers.setdefault(s.reviewer, []).append(s)

        for reviewer, items in reviewers.items():
            reviewer_total = sum(len(c.suggestion_texts) for c in items)
            lines.append(f"### @{reviewer} ({reviewer_total} suggestions)")
            lines.append("")

            idx = 1
            for c in items:
                for s_text in c.suggestion_texts:
                    lines.append(f"**{idx}. Line {c.original_line or '?'}** (`{c.path}`)")
                    lines.append("")
                    lines.append("Original:")
                    lines.append(fenced_block(c.original_text, "text"))
                    lines.append("")
                    lines.append("Suggested:")
                    lines.append(fenced_block(s_text, "text"))
                    lines.append("")
                    if c.comment_text and c.comment_type == "suggestion":
                        lines.append("Reviewer note:")
                        lines.append(c.comment_text.strip())
                        lines.append("")
                    if c.html_url:
                        lines.append(f"[View on GitHub]({c.html_url})")
                        lines.append("")
                    idx += 1

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
            if c.html_url:
                lines.append(f"[View on GitHub]({c.html_url})")
                lines.append("")

    return "\n".join(lines)


def format_diff_comparison(data: PRReviewData, max_file_chars: int | None = None) -> str:
    """Format for LLM paragraph-by-paragraph comparison."""

    lines: list[str] = []
    lines.append(f"# PR Style Analysis: {data.title}")
    lines.append(f"**PR:** {data.owner}/{data.repo}#{data.pr_number}")
    lines.append("")

    # Explicit suggestions
    suggestions = [c for c in data.comments if c.comment_type == "suggestion"]
    if suggestions:
        lines.append("## Explicit Suggestions (exact before → after)")
        lines.append("")

        k = 1
        for c in suggestions:
            for s_text in c.suggestion_texts:
                lines.append(f"### {k}. @{c.reviewer}")
                lines.append(f"**Path:** `{c.path}`  ")
                lines.append(f"**Line:** {c.original_line or '?'}")
                lines.append("")
                lines.append("**Before:**")
                lines.append(fenced_block(c.original_text, "text"))
                lines.append("")
                lines.append("**After:**")
                lines.append(fenced_block(s_text, "text"))
                lines.append("")
                if c.comment_text:
                    lines.append("**Reviewer note:**")
                    lines.append(c.comment_text.strip())
                    lines.append("")
                if c.html_url:
                    lines.append(f"[View on GitHub]({c.html_url})")
                    lines.append("")
                k += 1

    # Feedback
    feedback = [c for c in data.comments if c.comment_type == "feedback"]
    if feedback:
        lines.append("## Reviewer Feedback (requests without explicit replacement)")
        lines.append("")
        for i, c in enumerate(feedback, 1):
            lines.append(f"{i}. **@{c.reviewer}** ({c.path}:{c.original_line or '?'})")
            lines.append("")
            lines.append(c.comment_text)
            lines.append("")

    # File evolution
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
                lines.append(f"*File was renamed: {' → '.join(evo.all_paths)}*")
            lines.append("")

            if evo.first_content:
                lines.append(f"#### FIRST DRAFT ({evo.first_commit})")
                lines.append(
                    fenced_block(
                        maybe_truncate_text(evo.first_content, max_file_chars),
                        "markdown",
                    )
                )
                lines.append("")

            if evo.final_content:
                lines.append(f"#### FINAL VERSION ({evo.final_commit})")
                lines.append(
                    fenced_block(
                        maybe_truncate_text(evo.final_content, max_file_chars),
                        "markdown",
                    )
                )
                lines.append("")

    return "\n".join(lines)


# -----------------
# CLI
# -----------------


def parse_cli_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract GitHub PR review data")
    parser.add_argument("target", nargs="+", help="PR URL or 'owner/repo pr_number'")
    parser.add_argument("--diff", action="store_true", help="Include file evolution output")
    parser.add_argument("--json", action="store_true", help="Emit raw JSON instead of Markdown")
    parser.add_argument("--max-file-chars", type=int, default=None, help="Trim FIRST/FINAL dumps to this many characters per file")
    parser.add_argument("--no-files", action="store_true", help="Skip file evolution even when --diff is set")
    return parser.parse_args(argv)



def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    args = parse_cli_args(sys.argv[1:])

    if len(args.target) > 3:
        print("Error: Expected PR URL or 'owner/repo pr_number'", file=sys.stderr)
        sys.exit(1)

    try:
        owner, repo, pr_number = parse_pr_url(args.target)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    track_evolution = args.diff and not args.no_files

    try:
        data = fetch_pr_data(owner, repo, pr_number, track_evolution=track_evolution)
    except subprocess.CalledProcessError as e:
        print(f"Error fetching PR data: {e.stderr}", file=sys.stderr)
        sys.exit(1)

    if args.json:
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
        return

    if args.diff:
        print(format_diff_comparison(data, max_file_chars=args.max_file_chars))
        return

    print(format_suggestions_and_feedback(data))


if __name__ == "__main__":
    main()
