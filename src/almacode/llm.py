from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from almacode.config import AgentConfig


@dataclass(slots=True)
class LLMResponse:
    content: str
    raw: dict[str, Any]


class ModelLoadError(RuntimeError):
    pass


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

        try:
            self._client = Llama(**init_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(self._build_load_error(config, exc)) from exc
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

    @staticmethod
    def _build_load_error(config: AgentConfig, exc: Exception) -> str:
        model_path = Path(config.model_path).expanduser().resolve()
        message = str(exc).strip() or exc.__class__.__name__
        lower_name = model_path.name.lower()

        hints = [
            f"Failed to load GGUF model: {model_path}",
            f"llama.cpp error: {message}",
        ]

        if not model_path.exists():
            hints.append("The model path does not exist on disk.")
        elif model_path.is_dir():
            hints.append("The model path points to a directory, but a .gguf file is required.")

        if any(token in lower_name for token in ("-vl", "vl-", "vision", "multimodal")):
            hints.extend(
                [
                    "This looks like a vision-language model.",
                    "AlmaCode currently runs text-only chat completion and does not initialize multimodal chat handlers or mmproj weights.",
                    "For coding tasks, use a text model such as Qwen2.5-Coder-Instruct GGUF or another instruct/coder GGUF.",
                ]
            )
        else:
            hints.extend(
                [
                    "Make sure the file is a valid GGUF model supported by your installed llama.cpp build.",
                    "If the GGUF is older/newer than the bundled runtime expects, try a different model export or rebuild the binary from the latest release.",
                ]
            )

        return "\n".join(hints)
