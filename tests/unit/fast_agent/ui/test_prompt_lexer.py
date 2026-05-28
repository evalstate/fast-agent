from prompt_toolkit.document import Document

from fast_agent.ui.prompt.keybindings import ShellPrefixLexer


def test_hash_highlighting_only_applies_to_first_line() -> None:
    lexer = ShellPrefixLexer()
    document = Document("#agent hello\n# not a command here")

    tokens = lexer.lex_document(document)

    assert tokens(0) == [("class:comment-command", "#agent hello")]
    assert tokens(1) == [("", "# not a command here")]


def test_later_hash_lines_do_not_trigger_command_highlighting() -> None:
    lexer = ShellPrefixLexer()
    document = Document("plain text\n# heading")

    tokens = lexer.lex_document(document)

    assert tokens(0) == [("", "plain text")]
    assert tokens(1) == [("", "# heading")]


def test_shell_highlighting_only_applies_to_first_line() -> None:
    lexer = ShellPrefixLexer()
    document = Document("!echo hi\n!not-a-new-command")

    tokens = lexer.lex_document(document)

    assert tokens(0) == [("class:shell-command", "!echo hi")]
    assert tokens(1) == [("", "!not-a-new-command")]
