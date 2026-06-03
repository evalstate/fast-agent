from fast_agent.ui.command_payloads import CommandError, SkillsCommand
from fast_agent.ui.enhanced_prompt import parse_special_input


def test_parse_skills_defaults_to_list() -> None:
    result = parse_special_input("/skills")
    assert isinstance(result, SkillsCommand)
    assert result.action == "list"
    assert result.argument is None


def test_parse_skills_accepts_quoted_action_and_preserves_raw_argument() -> None:
    result = parse_special_input('/skills "add" "team skill" --registry local')
    assert isinstance(result, SkillsCommand)
    assert result.action == "add"
    assert result.argument == '"team skill" --registry local'


def test_parse_skills_action_matches_case_insensitively() -> None:
    result = parse_special_input("/SKILLS ADD team-skill")
    assert isinstance(result, SkillsCommand)
    assert result.action == "add"
    assert result.argument == "team-skill"


def test_parse_skills_reports_unclosed_quoted_action() -> None:
    result = parse_special_input('/skills "add')
    assert isinstance(result, CommandError)
    assert result.message == "Invalid /skills arguments: No closing quotation"
