from almacode.agent import parse_action


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

