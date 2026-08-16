"""Tests for URL elicitation required error handling helpers."""

from fast_agent.mcp.url_elicitation_required import parse_url_elicitation_required_data


class TestParseUrlElicitationRequiredData:
    def test_parses_valid_data(self) -> None:
        parsed = parse_url_elicitation_required_data(
            {
                "elicitations": [
                    {
                        "mode": "url",
                        "message": "Authorization is required.",
                        "url": "https://example.com/connect",
                        "elicitationId": "auth-001",
                    }
                ]
            }
        )

        assert parsed.issues == []
        assert len(parsed.elicitations) == 1
        assert parsed.elicitations[0].message == "Authorization is required."
        assert parsed.elicitations[0].url == "https://example.com/connect"
        assert parsed.elicitations[0].elicitation_id == "auth-001"

    def test_reports_missing_data(self) -> None:
        parsed = parse_url_elicitation_required_data(None)

        assert parsed.elicitations == []
        assert parsed.issues == ["error.data is missing"]

    def test_reports_non_list_elicitations(self) -> None:
        parsed = parse_url_elicitation_required_data({"elicitations": "not-a-list"})

        assert parsed.elicitations == []
        assert parsed.issues == ["error.data.elicitations must be a list, got str"]

    def test_reports_empty_elicitations_list(self) -> None:
        parsed = parse_url_elicitation_required_data({"elicitations": []})

        assert parsed.elicitations == []
        assert parsed.issues == ["error.data.elicitations is empty"]

    def test_collects_valid_and_invalid_entries(self) -> None:
        parsed = parse_url_elicitation_required_data(
            {
                "elicitations": [
                    {
                        "mode": "url",
                        "message": "Approve access",
                        "url": "https://example.com/ok",
                        "elicitationId": "ok-1",
                    },
                    {
                        "mode": "url",
                        "message": "Missing URL and elicitation id",
                    },
                    "not-an-object",
                ]
            }
        )

        assert len(parsed.elicitations) == 1
        assert parsed.elicitations[0].elicitation_id == "ok-1"
        assert len(parsed.issues) == 2
        assert "error.data.elicitations[1] is invalid" in parsed.issues[0]
        assert parsed.issues[1] == "error.data.elicitations[2] must be an object, got str"

    def test_accepts_snake_case_elicitation_id_with_non_compliant_issue(self) -> None:
        parsed = parse_url_elicitation_required_data(
            {
                "elicitations": [
                    {
                        "mode": "url",
                        "message": "Authorize",
                        "url": "https://example.com/auth",
                        "elicitation_id": "snake-1",
                    }
                ]
            }
        )

        assert len(parsed.elicitations) == 1
        assert parsed.elicitations[0].elicitation_id == "snake-1"
        assert len(parsed.issues) == 1
        assert "non-compliant" in parsed.issues[0]
        assert "elicitation_id" in parsed.issues[0]

    def test_trims_snake_case_elicitation_id_compatibility_value(self) -> None:
        parsed = parse_url_elicitation_required_data(
            {
                "elicitations": [
                    {
                        "mode": "url",
                        "message": "Authorize",
                        "url": "https://example.com/auth",
                        "elicitation_id": "  snake-1  ",
                    }
                ]
            }
        )

        assert len(parsed.elicitations) == 1
        assert parsed.elicitations[0].elicitation_id == "snake-1"
        assert len(parsed.issues) == 1
        assert "non-compliant" in parsed.issues[0]

    def test_rejects_blank_snake_case_elicitation_id(self) -> None:
        parsed = parse_url_elicitation_required_data(
            {
                "elicitations": [
                    {
                        "mode": "url",
                        "message": "Authorize",
                        "url": "https://example.com/auth",
                        "elicitation_id": "   ",
                    }
                ]
            }
        )

        assert parsed.elicitations == []
        assert len(parsed.issues) == 1
        assert "error.data.elicitations[0] is invalid" in parsed.issues[0]
