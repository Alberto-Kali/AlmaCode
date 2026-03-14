from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from almacode.config import AgentConfig


@dataclass(slots=True)
class LLMResponse:
    content: str
    raw: dict[str, Any]


class LlamaBackend:
    def __init__(self, config: AgentConfig) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install dependencies first."
            ) from exc

        init_kwargs: dict[str, Any] = {
            "model_path": str(Path(config.model_path).expanduser().resolve()),
            "n_ctx": config.n_ctx,
            "n_gpu_layers": config.n_gpu_layers,
            "verbose": config.verbose,
        }
        if config.chat_format:
            init_kwargs["chat_format"] = config.chat_format
        if config.n_threads is not None:
            init_kwargs["n_threads"] = config.n_threads

        self._client = Llama(**init_kwargs)
        self._config = config

    def complete(self, messages: list[dict[str, str]]) -> LLMResponse:
        response = self._client.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=self._config.temperature,
            top_p=self._config.top_p,
            max_tokens=self._config.max_tokens,
        )
        choice = response["choices"][0]["message"]["content"]
        if not isinstance(choice, str):
            choice = str(choice)
        return LLMResponse(content=choice, raw=response)

