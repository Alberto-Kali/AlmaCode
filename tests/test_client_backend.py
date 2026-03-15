import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from almacode.cli import build_parser, config_from_args
from almacode.config import AgentConfig, ClientSettings, autodetect_server_url, build_server_url, deprecated_runtime_flags, resolve_client_settings, save_client_settings
from almacode.llm import ContextOverflowError, LlamaBackend, ModelLoadError


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")
            return
        if self.path == "/props":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{\"default_generation_settings\":{\"n_ctx\":8192}}')
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        body = self.rfile.read(int(self.headers["Content-Length"]))
        payload = json.loads(body)
        reply = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "thought": "done",
                                "action": {"tool": "final_answer", "args": {"answer": payload["messages"][-1]["content"]}},
                            }
                        )
                    }
                }
            ]
        }
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(reply).encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


@pytest.fixture
def test_server() -> str:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_resolve_client_settings_from_file(tmp_path: Path) -> None:
    config_path = tmp_path / "client.json"
    config_path.write_text('{"server_url":"http://127.0.0.1:8080","api_key":"abc","request_timeout":30}', encoding="utf-8")
    settings = resolve_client_settings(None, None, None, None, None, config_path=config_path)
    assert settings.server_url == "http://127.0.0.1:8080"
    assert settings.api_key == "abc"
    assert settings.request_timeout == 30


def test_save_client_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "client.json"
    saved = save_client_settings(ClientSettings(server_url="http://127.0.0.1:8080"), config_path=config_path)
    assert saved == config_path
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    assert payload["server_url"] == "http://127.0.0.1:8080"


def test_deprecated_runtime_flags_are_detected() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--model", "/tmp/model.gguf", "hello"])
    active = deprecated_runtime_flags(args)
    assert active["model"] == "/tmp/model.gguf"


def test_config_from_args_rejects_deprecated_runtime_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--model", "/tmp/model.gguf", "hello"])
    with pytest.raises(ValueError):
        config_from_args(args)


def test_http_backend_completion(test_server: str, tmp_path: Path) -> None:
    config = AgentConfig(workspace=tmp_path, server_url=test_server)
    backend = LlamaBackend(config)
    assert config.context_window == 8192
    response = backend.complete([{"role": "user", "content": "hi"}])
    assert '"tool": "final_answer"' in response.content


def test_autodetect_server_url(test_server: str) -> None:
    detected = autodetect_server_url(candidates=[test_server])
    assert detected == test_server


def test_http_backend_health_failure(tmp_path: Path) -> None:
    config = AgentConfig(workspace=tmp_path, server_url="http://127.0.0.1:9")
    with pytest.raises(ModelLoadError):
        LlamaBackend(config)


def test_build_server_url_from_host_and_port() -> None:
    assert build_server_url(None, "127.0.0.1", 8080) == "http://127.0.0.1:8080"
    assert build_server_url(None, None, 9000) == "http://127.0.0.1:9000"


def test_config_from_args_accepts_server_host_and_port() -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--server-host", "127.0.0.1", "--server-port", "8080", "hello"])
    config = config_from_args(args)
    assert config.server_url == "http://127.0.0.1:8080"


def test_context_overflow_is_raised(tmp_path: Path) -> None:
    class OverflowHandler(_Handler):
        def do_POST(self) -> None:  # noqa: N802
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "error": {
                            "type": "exceed_context_size_error",
                            "n_prompt_tokens": 4133,
                            "n_ctx": 4096,
                        }
                    }
                ).encode("utf-8")
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), OverflowHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = AgentConfig(workspace=tmp_path, server_url=f"http://127.0.0.1:{server.server_port}")
        backend = LlamaBackend(config)
        with pytest.raises(ContextOverflowError):
            backend.complete([{"role": "user", "content": "hi"}])
    finally:
        server.shutdown()
        thread.join(timeout=2)
