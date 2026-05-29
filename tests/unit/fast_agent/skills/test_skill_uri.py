"""Direct tests for the `skill_uri` helpers.

Covers degenerate URI shapes that could otherwise compromise the reader's
allowed-URI-roots trust boundary if the loader ever forgets to validate.
"""

from __future__ import annotations

import pytest

from fast_agent.mcp.skill_uri import skill_name_from_uri, strip_skill_md


class TestStripSkillMd:
    def test_strips_trailing_skill_md(self) -> None:
        assert strip_skill_md("skill://acme/refunds/SKILL.md") == "skill://acme/refunds"

    def test_returns_unchanged_when_suffix_absent(self) -> None:
        assert strip_skill_md("skill://acme/refunds/README.md") == "skill://acme/refunds/README.md"

    def test_scheme_agnostic(self) -> None:
        assert strip_skill_md("github://o/r/s/SKILL.md") == "github://o/r/s"

    def test_degenerate_no_path_segment_strips_literally(self) -> None:
        # strip_skill_md is a pure lexical helper; it does NOT guard against
        # URIs without a skill-path segment. That guard lives at the loader.
        # The invariant callers rely on is: if strip_skill_md is given a URI
        # that skill_name_from_uri accepts as non-empty, the result is a
        # well-formed root.
        assert strip_skill_md("skill://SKILL.md") == "skill:/"

    def test_tolerates_trailing_slash(self) -> None:
        # A buggy server publishing `.../SKILL.md/` (non-conformant but
        # possible) must not seed a root that includes the SKILL.md segment
        # into the reader's allow-list.
        assert strip_skill_md("skill://acme/refunds/SKILL.md/") == "skill://acme/refunds"
        assert skill_name_from_uri("skill://acme/refunds/SKILL.md/") == "refunds"


class TestSkillNameFromUri:
    def test_single_segment(self) -> None:
        assert skill_name_from_uri("skill://git-workflow/SKILL.md") == "git-workflow"

    def test_nested_segments_returns_final(self) -> None:
        assert (
            skill_name_from_uri("github://owner/repo/skills/refunds/SKILL.md") == "refunds"
        )

    def test_returns_none_when_suffix_absent(self) -> None:
        assert skill_name_from_uri("skill://acme/refunds/README.md") is None

    @pytest.mark.parametrize(
        "uri",
        [
            "skill://SKILL.md",  # no path segment between :// and /SKILL.md
            "skill:///SKILL.md",  # empty first segment
        ],
    )
    def test_returns_none_for_degenerate_shapes(self, uri: str) -> None:
        # This is the contract the loader depends on: no skill-path segment
        # means no registrable manifest. Returning "" here instead of None
        # previously let `skill://SKILL.md` slip past the `if url_name and ...`
        # check and seed `skill:/` into the reader's allowed-roots set.
        assert skill_name_from_uri(uri) is None
