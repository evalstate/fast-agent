from __future__ import annotations

from fast_agent.commands.metadata_labels import metadata_argument_label, metadata_option_label


def test_metadata_argument_label_formats_optional_value_name() -> None:
    assert metadata_argument_label({"name": " target ", "value_name": " latest|id "}) == (
        "`target` (`latest|id`)"
    )


def test_metadata_argument_label_omits_blank_name() -> None:
    assert metadata_argument_label({"name": "   ", "value_name": "path"}) is None


def test_metadata_option_label_formats_value_name_and_aliases() -> None:
    assert metadata_option_label(
        {
            "name": " --output ",
            "value_name": " path ",
            "aliases": [" -o "],
        }
    ) == "`--output path`, `-o`"


def test_metadata_option_label_deduplicates_alias_labels_case_insensitively() -> None:
    assert metadata_option_label(
        {
            "name": "--output",
            "value_name": None,
            "aliases": [" --output ", "--OUTPUT", "-o"],
        }
    ) == "`--output`, `-o`"


def test_metadata_option_label_escapes_backticks_in_code_spans() -> None:
    assert metadata_option_label(
        {
            "name": "--output`path",
            "value_name": "file`path",
            "aliases": ["-o`"],
        }
    ) == "`` --output`path file`path ``, `` -o` ``"
