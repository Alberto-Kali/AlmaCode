from pathlib import Path
from urllib import request

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


def test_open_url_extracts_text(monkeypatch: pytest.MonkeyPatch, tools: WorkspaceTools) -> None:
    class _Response:
        headers = {"Content-Type": "text/html; charset=utf-8"}

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return b"<html><body><h1>Hello</h1><p>World</p></body></html>"

    monkeypatch.setattr(request, "urlopen", lambda req, timeout=20: _Response())
    payload = tools.open_url("https://example.com")
    assert "Hello" in payload["text"]
    assert "World" in payload["text"]


def test_web_search_parses_results(monkeypatch: pytest.MonkeyPatch, tools: WorkspaceTools) -> None:
    html = """
    <html><body>
    <a class="result__a" href="https://example.com/one">Result One</a>
    <a class="result__a" href="https://example.com/two">Result Two</a>
    </body></html>
    """

    class _Response:
        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def read(self) -> bytes:
            return html.encode("utf-8")

    monkeypatch.setattr(request, "urlopen", lambda req, timeout=20: _Response())
    payload = tools.web_search("test", limit=2)
    assert payload["results"][0]["title"] == "Result One"
    assert payload["results"][1]["url"] == "https://example.com/two"


def test_run_command_prefers_detected_virtualenv(tools: WorkspaceTools) -> None:
    venv_dir = tools.resolve_path("fastapi-env")
    bin_dir = venv_dir / "bin"
    bin_dir.mkdir(parents=True)
    (venv_dir / "pyvenv.cfg").write_text("home = /usr/bin\n", encoding="utf-8")
    python_path = bin_dir / "python"
    python_path.write_text("#!/bin/sh\nprintf 'venv-python\\n'\n", encoding="utf-8")
    python_path.chmod(0o755)

    result = tools.run_command("python", timeout=5)
    assert result["exit_code"] == 0
    assert result["stdout"].strip() == "venv-python"
    assert result["virtual_env"] == str(venv_dir)
