from pathlib import Path

from almacode.agent import build_user_message, parse_action
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
    assert "--mmproj" in message


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


def test_autodetect_qwen25_vl_handler() -> None:
    handler = LlamaBackend._autodetect_handler_name(Path("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"))
    assert handler == "qwen2.5-vl"


def test_detect_unsupported_qwen3_vl_model() -> None:
    message = LlamaBackend._detect_unsupported_multimodal_model(Path("Qwen3VL-8B-Thinking-Q8_0.gguf"))
    assert message is not None
    assert "Qwen3-VL" in message
    assert "llama-server" in message
