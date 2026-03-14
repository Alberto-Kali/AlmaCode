from __future__ import annotations

import json
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib import error, request

from almacode.config import AgentConfig


class ServerBackendError(RuntimeError):
    pass


class LlamaServerBackend:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._port = self._find_free_port()
        self._process: subprocess.Popen[str] | None = None
        self._binary = self._resolve_binary(config)
        self._start_server()

    def complete(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        payload = {
            "model": str(Path(self._config.model_path).expanduser().resolve()),
            "messages": messages,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "max_tokens": self._config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        return self._post_json("/v1/chat/completions", payload)

    def close(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            return int(sock.getsockname()[1])

    @staticmethod
    def _resolve_binary(config: AgentConfig) -> str:
        candidates: list[str] = []
        if config.llama_server_binary:
            candidates.append(str(Path(config.llama_server_binary).expanduser().resolve()))
        candidates.extend(
            [
                "llama-server",
                "llama-server.exe",
                "llama-cli",
                "llama-cli.exe",
            ]
        )

        for candidate in candidates:
            resolved = shutil.which(candidate) if not Path(candidate).is_absolute() else candidate
            if resolved and Path(resolved).exists():
                return str(resolved)

        raise ServerBackendError(
            "Could not find a llama.cpp server binary.\n"
            "Pass --llama-server-binary /path/to/llama-server or install llama.cpp so `llama-server` is on PATH."
        )

    def _build_command(self) -> list[str]:
        command = [
            self._binary,
            "--model",
            str(Path(self._config.model_path).expanduser().resolve()),
            "--host",
            "127.0.0.1",
            "--port",
            str(self._port),
            "--ctx-size",
            str(self._config.n_ctx),
            "--jinja",
        ]
        if self._config.mmproj_path is not None:
            command.extend(["--mmproj", str(Path(self._config.mmproj_path).expanduser().resolve())])
        if self._config.n_gpu_layers != 0:
            command.extend(["--gpu-layers", str(self._config.n_gpu_layers)])
        if self._config.n_threads is not None:
            command.extend(["--threads", str(self._config.n_threads)])
        if self._config.chat_format:
            command.extend(["--chat-template", self._config.chat_format])
        return command

    def _start_server(self) -> None:
        command = self._build_command()
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        deadline = time.time() + 30
        last_error = ""
        while time.time() < deadline:
            if self._process.poll() is not None:
                stderr = ""
                if self._process.stderr is not None:
                    stderr = self._process.stderr.read().strip()
                raise ServerBackendError(
                    "llama-server exited during startup.\n"
                    f"Command: {' '.join(command)}\n"
                    f"stderr: {stderr or '[no stderr output]'}"
                )

            try:
                self._get("/health")
                return
            except ServerBackendError as exc:
                last_error = str(exc)
                time.sleep(0.4)

        raise ServerBackendError(f"Timed out waiting for llama-server to become ready. Last error: {last_error}")

    def _get(self, path: str) -> dict[str, Any]:
        url = f"http://127.0.0.1:{self._port}{path}"
        try:
            with request.urlopen(url, timeout=3) as response:
                data = response.read().decode("utf-8")
                return json.loads(data) if data else {}
        except error.URLError as exc:
            raise ServerBackendError(str(exc)) from exc

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"http://127.0.0.1:{self._port}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with request.urlopen(req, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            payload = exc.read().decode("utf-8", errors="replace")
            raise ServerBackendError(f"HTTP {exc.code} from llama-server: {payload}") from exc
        except error.URLError as exc:
            raise ServerBackendError(f"Failed to contact llama-server: {exc}") from exc
