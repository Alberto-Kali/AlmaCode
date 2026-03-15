from __future__ import annotations

import json
import time
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


@dataclass(slots=True)
class ContextOverflowError(ModelLoadError):
    prompt_tokens: int
    context_window: int
    details: str


@dataclass(slots=True)
class TransientServerError(ModelLoadError):
    status_code: int
    details: str


class LlamaBackend:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._healthcheck()
        self._hydrate_server_limits()

    def complete(self, messages: list[dict[str, Any]]) -> LLMResponse:
        return self.chat(messages, response_format={"type": "json_object"})

    def summarize(self, messages: list[dict[str, Any]], max_tokens: int) -> str:
        response = self.chat(messages, max_tokens=max_tokens, response_format=None)
        return response.content

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload = {
            "messages": messages,
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "max_tokens": max_tokens or self._config.max_tokens,
        }
        if response_format is not None:
            payload["response_format"] = response_format
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

    def _hydrate_server_limits(self) -> None:
        try:
            props = self._get_json("/props")
        except ModelLoadError:
            return
        n_ctx = props.get("default_generation_settings", {}).get("n_ctx")
        if isinstance(n_ctx, int) and n_ctx > 0:
            self._config.context_window = n_ctx

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
        last_error: Exception | None = None
        for attempt in range(self._config.request_retries + 1):
            try:
                with request.urlopen(req, timeout=self._config.request_timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code == 400:
                    try:
                        payload = json.loads(body)
                    except json.JSONDecodeError:
                        payload = None
                    if isinstance(payload, dict):
                        error_payload = payload.get("error", {})
                        if error_payload.get("type") == "exceed_context_size_error":
                            raise ContextOverflowError(
                                prompt_tokens=int(error_payload.get("n_prompt_tokens", 0)),
                                context_window=int(error_payload.get("n_ctx", self._config.context_window)),
                                details=body,
                            ) from exc
                if exc.code in {502, 503, 504} and attempt < self._config.request_retries:
                    last_error = TransientServerError(status_code=exc.code, details=body)
                    time.sleep(0.6 * (attempt + 1))
                    continue
                raise ModelLoadError(f"HTTP {exc.code} from llama-server {path}: {body}") from exc
            except error.URLError as exc:
                if attempt < self._config.request_retries:
                    last_error = exc
                    time.sleep(0.6 * (attempt + 1))
                    continue
                raise ModelLoadError(f"Failed to contact llama-server {path}: {exc}") from exc
        if last_error:
            raise ModelLoadError(f"Failed to contact llama-server {path}: {last_error}")
        raise ModelLoadError(f"Failed to contact llama-server {path}")

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers
