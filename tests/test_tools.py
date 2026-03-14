from pathlib import Path

import pytest

from almacode.tools import ToolError, WorkspaceTools


@pytest.fixture
def tools(tmp_path: Path) -> WorkspaceTools:
    return WorkspaceTools(root=tmp_path, default_timeout=5)


def test_write_and_read_file(tools: WorkspaceTools) -> None:
    tools.write_file("notes.txt", "alpha\nbeta\n")
    content = tools.read_file("notes.txt")
    assert "1: alpha" in content
    assert "2: beta" in content


def test_replace_in_file(tools: WorkspaceTools) -> None:
    tools.write_file("notes.txt", "one two one")
    result = tools.replace_in_file("notes.txt", "one", "zero", count=1)
    assert result["replacements"] == 1
    assert "zero two one" in tools.resolve_path("notes.txt").read_text(encoding="utf-8")


def test_replace_in_file_count_zero_replaces_all(tools: WorkspaceTools) -> None:
    tools.write_file("notes.txt", "one two one")
    result = tools.replace_in_file("notes.txt", "one", "zero")
    assert result["replacements"] == 2
    assert tools.resolve_path("notes.txt").read_text(encoding="utf-8") == "zero two zero"


def test_path_escape_is_blocked(tools: WorkspaceTools) -> None:
    with pytest.raises(ToolError):
        tools.read_file("../secret.txt")
