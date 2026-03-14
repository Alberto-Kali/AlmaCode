from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from almacode.config import AgentConfig
from almacode.server_backend import LlamaServerBackend, ServerBackendError


SUPPORTED_MM_HANDLERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "llava-1-5": ("Llava15ChatHandler", ("llava-v1.5", "llava15")),
    "llava-1-6": ("Llava16ChatHandler", ("llava-v1.6", "llava16")),
    "moondream2": ("MoondreamChatHandler", ("moondream",)),
    "nanollava": ("NanollavaChatHandler", ("nano-llava",)),
    "llama-3-vision-alpha": ("Llama3VisionAlphaChatHandler", ("llama3-vision", "vision-alpha")),
    "minicpm-v-2.6": ("MiniCPMv26ChatHandler", ("minicpm", "minicpm-v26")),
    "qwen2.5-vl": ("Qwen25VLChatHandler", ("qwen25-vl", "qwen2-vl", "qwen2.5vl")),
}

MMPROJ_PARAM_NAMES = ("clip_model_path", "mmproj_path", "proj_model_path", "model_path")


@dataclass(slots=True)
class LLMResponse:
    content: str
    raw: dict[str, Any]


class ModelLoadError(RuntimeError):
    pass


class LlamaBackend:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config
        self._server_backend: LlamaServerBackend | None = None

        unsupported_reason = self._detect_unsupported_multimodal_model(Path(config.model_path))
        if config.backend == "server":
            self._init_server_backend(config)
            return

        if unsupported_reason is not None:
            if config.mmproj_path is not None:
                try:
                    self._init_server_backend(config)
                    return
                except ModelLoadError as exc:
                    raise ModelLoadError(f"{unsupported_reason}\n\nServer fallback failed:\n{exc}") from exc
            raise ModelLoadError(unsupported_reason)

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

        chat_handler = self._build_chat_handler(config)
        if chat_handler is not None:
            init_kwargs["chat_handler"] = chat_handler

        try:
            self._client = Llama(**init_kwargs)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(self._build_load_error(config, exc)) from exc

    def complete(self, messages: list[dict[str, Any]]) -> LLMResponse:
        if self._server_backend is not None:
            response = self._server_backend.complete(messages)
            choice = response["choices"][0]["message"]["content"]
            if not isinstance(choice, str):
                choice = str(choice)
            return LLMResponse(content=choice, raw=response)

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

    def close(self) -> None:
        if self._server_backend is not None:
            self._server_backend.close()

    def _init_server_backend(self, config: AgentConfig) -> None:
        try:
            self._server_backend = LlamaServerBackend(config)
        except ServerBackendError as exc:
            raise ModelLoadError(str(exc)) from exc

    @staticmethod
    def _normalize_handler_name(name: str) -> str:
        lowered = name.strip().lower()
        if lowered in SUPPORTED_MM_HANDLERS:
            return lowered
        for canonical, (_, aliases) in SUPPORTED_MM_HANDLERS.items():
            if lowered in aliases:
                return canonical
        raise ModelLoadError(
            "Unsupported multimodal handler: "
            f"{name}. Supported values: {', '.join(sorted(SUPPORTED_MM_HANDLERS))}"
        )

    @classmethod
    def _autodetect_handler_name(cls, model_path: Path) -> str | None:
        lowered = model_path.name.lower()
        if "qwen2.5" in lowered and "vl" in lowered:
            return "qwen2.5-vl"
        if "llava" in lowered and ("1.6" in lowered or "v1.6" in lowered):
            return "llava-1-6"
        if "llava" in lowered and ("1.5" in lowered or "v1.5" in lowered):
            return "llava-1-5"
        if "moondream2" in lowered:
            return "moondream2"
        if "nanollava" in lowered:
            return "nanollava"
        if "llama-3" in lowered and "vision" in lowered:
            return "llama-3-vision-alpha"
        if "minicpm" in lowered and "v-2.6" in lowered:
            return "minicpm-v-2.6"
        return None

    @staticmethod
    def _detect_unsupported_multimodal_model(model_path: Path) -> str | None:
        lowered = model_path.name.lower()
        if "qwen3" in lowered and "vl" in lowered:
            return "\n".join(
                [
                    f"Failed to load GGUF model: {model_path.expanduser().resolve()}",
                    "This looks like a Qwen3-VL model.",
                    "The direct Python llama-cpp backend used by AlmaCode does not currently provide a dedicated Qwen3-VL chat handler.",
                    "AlmaCode can still try an external llama.cpp server fallback if you provide --mmproj and a usable llama-server binary.",
                    "Pass --llama-server-binary /path/to/llama-server, or put llama-server on PATH.",
                ]
            )
        return None

    @classmethod
    def _build_chat_handler(cls, config: AgentConfig) -> Any | None:
        explicit_handler = cls._normalize_handler_name(config.mm_handler) if config.mm_handler else None
        auto_handler = cls._autodetect_handler_name(Path(config.model_path))
        handler_name = explicit_handler or auto_handler
        if handler_name is None:
            return None

        if config.mmproj_path is None:
            raise ModelLoadError(
                "This model looks multimodal, but no mmproj path was provided.\n"
                "Pass --mmproj /path/to/mmproj.gguf.\n"
                f"Detected handler: {handler_name}"
            )

        try:
            from llama_cpp import llama_chat_format
        except ImportError as exc:
            raise ModelLoadError(
                "llama-cpp-python does not expose llama_chat_format, so multimodal handlers are unavailable."
            ) from exc

        class_name = SUPPORTED_MM_HANDLERS[handler_name][0]
        try:
            handler_cls = getattr(llama_chat_format, class_name)
        except AttributeError as exc:
            raise ModelLoadError(
                f"Installed llama-cpp-python does not provide {class_name}. "
                "Upgrade to a version that supports this multimodal handler."
            ) from exc

        mmproj_path = str(Path(config.mmproj_path).expanduser().resolve())
        signature = inspect.signature(handler_cls)
        kwargs: dict[str, Any] = {}
        for param_name in MMPROJ_PARAM_NAMES:
            if param_name in signature.parameters:
                kwargs[param_name] = mmproj_path
                break

        if kwargs:
            return handler_cls(**kwargs)

        required_params = [
            parameter
            for parameter in signature.parameters.values()
            if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            and parameter.default is inspect._empty
            and parameter.name != "self"
        ]
        if len(required_params) == 1:
            return handler_cls(mmproj_path)

        raise ModelLoadError(
            f"Unable to initialize {class_name}: unsupported constructor signature {signature}"
        )

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
                    "Multimodal models require a matching mmproj file and a supported chat handler.",
                    "Pass --mmproj /path/to/mmproj.gguf and optionally --mm-handler qwen2.5-vl (or another supported handler).",
                    "If the Python backend still fails, try --backend server with --llama-server-binary /path/to/llama-server.",
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
