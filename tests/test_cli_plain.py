import json

from almacode.cli import format_plain_tool_event


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
