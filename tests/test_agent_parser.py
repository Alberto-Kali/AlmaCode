from pathlib import Path

from almacode.agent import build_user_message, parse_action


def test_parse_action_accepts_nested_schema() -> None:
    action = parse_action(
        '{"thought":"inspect source","action":{"tool":"read_file","args":{"path":"src/app.py"}}}'
    )
    assert action.tool == "read_file"
    assert action.args["path"] == "src/app.py"


def test_parse_action_extracts_embedded_json() -> None:
    action = parse_action(
        '```json\n{"thought":"done","action":{"tool":"final_answer","args":{"answer":"ok"}}}\n```'
    )
    assert action.tool == "final_answer"
    assert action.args["answer"] == "ok"


def test_build_user_message_without_images() -> None:
    message = build_user_message("describe")
    assert message == {"role": "user", "content": "describe"}


def test_build_user_message_with_local_image(tmp_path: Path) -> None:
    image_path = tmp_path / "test.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    message = build_user_message("describe", [str(image_path)])
    content = message["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
