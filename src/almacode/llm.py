from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from almacode.config import AgentConfig


@dataclass(slots=True)
class LLMResponse:
    content: str
    raw: dict[str, Any]


class ModelLoadError(RuntimeError):
    pass


class LlamaBackend:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._healthcheck()

    def complete(self, messages: list[dict[str, Any]]) -> LLMResponse:
        payload = {
            "messages": messages,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "max_tokens": self._config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        response = self._post_json("/v1/chat/completions", payload)
        try:
            choice = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelLoadError(f"Unexpected response from llama-server: {response}") from exc
        if not isinstance(choice, str):
            choice = str(choice)
        return LLMResponse(content=choice, raw=response)

    def _healthcheck(self) -> None:
        paths = ["/health", "/v1/models"]
        for path in paths:
            try:
                self._get_json(path)
                return
            except ModelLoadError:
                continue
        raise ModelLoadError(
            f"Could not reach llama-server at {self._config.server_url}.\n"
            "Make sure the server is running and reachable from this machine."
        )

    def _get_json(self, path: str) -> dict[str, Any]:
        url = f"{self._config.server_url}{path}"
        req = request.Request(url, headers=self._headers())
        try:
            with request.urlopen(req, timeout=self._config.request_timeout) as response:
                raw = response.read().decode("utf-8")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ModelLoadError(f"HTTP {exc.code} from llama-server {path}: {body}") from exc
        except error.URLError as exc:
            raise ModelLoadError(f"Failed to contact llama-server {path}: {exc}") from exc
        return json.loads(raw) if raw else {}

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._config.server_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url,
            data=body,
            headers=self._headers() | {"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self._config.request_timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise ModelLoadError(f"HTTP {exc.code} from llama-server {path}: {body}") from exc
        except error.URLError as exc:
            raise ModelLoadError(f"Failed to contact llama-server {path}: {exc}") from exc

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers
