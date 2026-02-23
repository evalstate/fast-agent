from pathlib import Path

from fast_agent.core.agent_card_validation import scan_agent_card_directory, scan_agent_card_path


def test_scan_agent_cards_reports_invalid_history_json(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    history_path.write_text('{"messages": ["bad\x00"]}', encoding="utf-8")

    card_path = tmp_path / "agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: test_agent",
                "messages:",
                "  - history.json",
            ]
        ),
        encoding="utf-8",
    )

    results = scan_agent_card_directory(tmp_path)
    assert len(results) == 1

    errors = results[0].errors
    assert any("History file failed to load" in err for err in errors)
    assert any("Failed to parse JSON prompt file" in err for err in errors)


def test_scan_agent_card_path_for_file(tmp_path: Path) -> None:
    card_path = tmp_path / "agent.yaml"
    card_path.write_text(
        "\n".join([
            "name: solo_agent",
            "model: gpt-4.1",
        ]),
        encoding="utf-8",
    )

    results = scan_agent_card_path(card_path)
    assert len(results) == 1
    assert results[0].name == "solo_agent"
    assert results[0].path == card_path


def test_scan_agent_card_path_for_directory(tmp_path: Path) -> None:
    card_path = tmp_path / "agent.yaml"
    card_path.write_text(
        "\n".join([
            "name: dir_agent",
            "model: gpt-4.1",
        ]),
        encoding="utf-8",
    )

    results = scan_agent_card_path(tmp_path)
    assert len(results) == 1
    assert results[0].name == "dir_agent"


def test_scan_agent_cards_reports_dependency_cycle(tmp_path: Path) -> None:
    agent_a = tmp_path / "agent_a.yaml"
    agent_a.write_text(
        "\n".join(
            [
                "name: agent_a",
                "agents:",
                "  - agent_b",
            ]
        ),
        encoding="utf-8",
    )
    agent_b = tmp_path / "agent_b.yaml"
    agent_b.write_text(
        "\n".join(
            [
                "name: agent_b",
                "agents:",
                "  - agent_a",
            ]
        ),
        encoding="utf-8",
    )

    results = scan_agent_card_directory(tmp_path)
    errors_by_name = {entry.name: entry.errors for entry in results}

    assert any(
        "Circular dependency detected" in err for err in errors_by_name.get("agent_a", [])
    )
    assert any(
        "Circular dependency detected" in err for err in errors_by_name.get("agent_b", [])
    )


def test_scan_agent_cards_allows_acyclic_dependencies(tmp_path: Path) -> None:
    agent_a = tmp_path / "agent_a.yaml"
    agent_a.write_text(
        "\n".join(
            [
                "name: agent_a",
                "agents:",
                "  - agent_b",
            ]
        ),
        encoding="utf-8",
    )
    agent_b = tmp_path / "agent_b.yaml"
    agent_b.write_text("name: agent_b\n", encoding="utf-8")

    results = scan_agent_card_directory(tmp_path)

    for entry in results:
        assert not any("Circular dependency detected" in err for err in entry.errors)


def test_scan_agent_cards_reports_invalid_mcp_connect_target(tmp_path: Path) -> None:
    card_path = tmp_path / "agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: mcp_agent",
                "mcp_connect:",
                "  - target: ''",
            ]
        ),
        encoding="utf-8",
    )

    results = scan_agent_card_directory(tmp_path)
    assert len(results) == 1
    assert any("mcp_connect[0].target" in err for err in results[0].errors)


def test_scan_agent_cards_reports_unparseable_mcp_connect_entry(tmp_path: Path) -> None:
    card_path = tmp_path / "agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: mcp_agent",
                "mcp_connect:",
                "  - target: \"npx 'unterminated\"",
            ]
        ),
        encoding="utf-8",
    )

    results = scan_agent_card_directory(tmp_path)
    assert len(results) == 1
    assert any("Invalid mcp_connect target" in err for err in results[0].errors)


def test_scan_agent_cards_rejects_mcp_connect_url_with_embedded_auth_flag(tmp_path: Path) -> None:
    card_path = tmp_path / "agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: mcp_agent",
                "mcp_connect:",
                '  - target: "https://demo.hf.space --auth token"',
            ]
        ),
        encoding="utf-8",
    )

    results = scan_agent_card_directory(tmp_path)
    assert len(results) == 1
    assert any("pure target string" in err for err in results[0].errors)
    assert any("--auth" in err for err in results[0].errors)


def test_scan_agent_cards_reports_missing_shell_cwd(tmp_path: Path) -> None:
    missing_cwd = tmp_path / "missing-shell-cwd"
    card_path = tmp_path / "agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: shell_agent",
                f"cwd: {missing_cwd}",
            ]
        ),
        encoding="utf-8",
    )

    results = scan_agent_card_directory(tmp_path)
    assert len(results) == 1
    assert any("Shell cwd does not exist" in err for err in results[0].errors)


def test_scan_agent_cards_reports_shell_cwd_when_path_is_file(tmp_path: Path) -> None:
    shell_file = tmp_path / "not-a-directory.txt"
    shell_file.write_text("x", encoding="utf-8")
    card_path = tmp_path / "agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: shell_agent",
                f"cwd: {shell_file}",
            ]
        ),
        encoding="utf-8",
    )

    results = scan_agent_card_directory(tmp_path)
    assert len(results) == 1
    assert any("Shell cwd is not a directory" in err for err in results[0].errors)


def test_scan_agent_cards_reports_invalid_shell_cwd_type(tmp_path: Path) -> None:
    card_path = tmp_path / "agent.yaml"
    card_path.write_text(
        "\n".join(
            [
                "name: shell_agent",
                "cwd:",
                "  nested: value",
            ]
        ),
        encoding="utf-8",
    )

    results = scan_agent_card_directory(tmp_path)
    assert len(results) == 1
    assert "'cwd' must be a string" in results[0].errors
