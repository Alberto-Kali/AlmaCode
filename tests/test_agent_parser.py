from almacode.agent import parse_action
from almacode.config import AgentConfig
from almacode.llm import LlamaBackend


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


def test_model_load_error_mentions_vl_models() -> None:
    config = AgentConfig(model_path="Qwen3VL-8B-Thinking-Q8_0.gguf", workspace=".")
    message = LlamaBackend._build_load_error(config, ValueError("Failed to load model from file"))
    assert "vision-language model" in message
    assert "text-only" in message
