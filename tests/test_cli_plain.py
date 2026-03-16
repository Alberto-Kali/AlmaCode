import json

from almacode.cli import format_plain_tool_event
from almacode.prompts import build_tool_feedback


def test_format_plain_tool_event_includes_command_output() -> None:
    payload = json.dumps(
        {
            "command": "python -m pip install fastapi",
            "cwd": ".",
            "exit_code": 1,
            "stdout": "Collecting fastapi",
            "stderr": "ssl error",
            "virtual_env": "/tmp/project/.venv",
        },
        ensure_ascii=True,
        indent=2,
    )
    message = "\n".join(
        [
            "tool: run_command",
            "command: python -m pip install fastapi",
            "cwd: .",
            "result:",
            payload,
        ]
    )
    rendered = format_plain_tool_event("tool", message)
    assert any("Command done" in line for line in rendered)
    assert any("stdout:" in line for line in rendered)
    assert any("Collecting fastapi" in line for line in rendered)
    assert any("stderr:" in line for line in rendered)
    assert any("/tmp/project/.venv" in line for line in rendered)


def test_build_tool_feedback_marks_command_success_even_with_stderr() -> None:
    payload = json.dumps(
        {
            "command": "pip install fastapi uvicorn",
            "cwd": ".",
            "exit_code": 0,
            "stdout": "Successfully installed fastapi uvicorn",
            "stderr": "A new release of pip is available",
            "virtual_env": "/tmp/project/.venv",
        },
        ensure_ascii=True,
        indent=2,
    )
    feedback = build_tool_feedback("run_command", payload)
    assert "status: success" in feedback
    assert "interpretation: the command completed successfully" in feedback
    assert "stderr_note: stderr contains warnings/notices but the command still succeeded" in feedback
    assert "Successfully installed fastapi uvicorn" in feedback
