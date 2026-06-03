from fast_agent.tools.edit_file_tool import EditFileInput, extract_edit_file_input


def test_extract_edit_file_input_accepts_valid_arguments() -> None:
    assert extract_edit_file_input(
        {
            "path": " notes.txt ",
            "old_string": "before",
            "new_string": "",
            "replace_all": True,
        }
    ) == EditFileInput(
        path="notes.txt",
        old_string="before",
        new_string="",
        replace_all=True,
    )


def test_extract_edit_file_input_rejects_invalid_arguments() -> None:
    invalid_arguments = [
        None,
        {},
        {"path": "", "old_string": "before", "new_string": "after"},
        {"path": "notes.txt", "old_string": 1, "new_string": "after"},
        {"path": "notes.txt", "old_string": "before", "new_string": None},
        {
            "path": "notes.txt",
            "old_string": "before",
            "new_string": "after",
            "replace_all": "yes",
        },
    ]

    for arguments in invalid_arguments:
        assert extract_edit_file_input(arguments) is None


def test_extract_edit_file_input_rejects_non_dict_arguments() -> None:
    assert extract_edit_file_input("not a mapping") is None  # ty: ignore[invalid-argument-type]
