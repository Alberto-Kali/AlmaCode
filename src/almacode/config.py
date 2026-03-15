from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


CLIENT_CONFIG_PATH = Path.home() / ".config" / "almacode" / "client.json"
DEFAULT_SERVER_CANDIDATES = [
    "http://127.0.0.1:8080",
    "http://localhost:8080",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]


@dataclass(slots=True)
class ClientSettings:
    server_url: str
    api_key: str | None = None
    request_timeout: int = 120


@dataclass(slots=True)
class AgentConfig:
    workspace: Path
    server_url: str
    api_key: str | None = None
    request_timeout: int = 120
    max_steps: int = 18
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 768
    command_timeout: int = 60
    verbose: bool = False
    system_note: str = ""


def load_client_settings(config_path: Path = CLIENT_CONFIG_PATH) -> ClientSettings | None:
    if not config_path.exists():
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    server_url = str(payload.get("server_url", "")).strip()
    if not server_url:
        return None
    api_key = payload.get("api_key")
    if api_key is not None:
        api_key = str(api_key)
    request_timeout = int(payload.get("request_timeout", 120))
    return ClientSettings(server_url=server_url, api_key=api_key, request_timeout=request_timeout)


def save_client_settings(settings: ClientSettings, config_path: Path = CLIENT_CONFIG_PATH, *, overwrite: bool = False) -> Path:
    config_path = config_path.expanduser().resolve()
    if config_path.exists() and not overwrite:
        raise ValueError(f"Client config already exists: {config_path}")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "server_url": settings.server_url,
        "api_key": settings.api_key,
        "request_timeout": settings.request_timeout,
    }
    config_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    return config_path


def _server_looks_alive(server_url: str, timeout: int) -> bool:
    for path in ("/health", "/v1/models"):
        req = request.Request(f"{server_url.rstrip('/')}{path}", headers={"Accept": "application/json"})
        try:
            with request.urlopen(req, timeout=timeout) as response:
                if 200 <= response.status < 300:
                    return True
        except (error.URLError, error.HTTPError):
            continue
    return False


def autodetect_server_url(timeout: int = 2, candidates: list[str] | None = None) -> str | None:
    for server_url in candidates or DEFAULT_SERVER_CANDIDATES:
        if _server_looks_alive(server_url, timeout):
            return server_url.rstrip("/")
    return None


def resolve_client_settings(
    server_url: str | None,
    api_key: str | None,
    request_timeout: int | None,
    config_path: Path = CLIENT_CONFIG_PATH,
) -> ClientSettings:
    file_settings = load_client_settings(config_path)
    resolved_server_url = (
        (server_url or "").strip()
        or os.environ.get("ALMACODE_SERVER_URL", "").strip()
        or (file_settings.server_url if file_settings else "")
        or (autodetect_server_url(timeout=request_timeout or (file_settings.request_timeout if file_settings else 2)) or "")
    )
    resolved_api_key = api_key or os.environ.get("ALMACODE_API_KEY") or (file_settings.api_key if file_settings else None)
    resolved_request_timeout = request_timeout or (file_settings.request_timeout if file_settings else 120)

    if not resolved_server_url:
        raise ValueError(
            "No server URL configured.\n"
            "Pass --server-url, set ALMACODE_SERVER_URL, or create ~/.config/almacode/client.json."
        )

    return ClientSettings(
        server_url=resolved_server_url.rstrip("/"),
        api_key=resolved_api_key,
        request_timeout=resolved_request_timeout,
    )


def deprecated_runtime_flags(args: Any) -> dict[str, Any]:
    tracked_flags = {
        "model": getattr(args, "model", None),
        "mmproj": getattr(args, "mmproj", None),
        "mm_handler": getattr(args, "mm_handler", None),
        "backend": getattr(args, "backend", None),
        "llama_server_binary": getattr(args, "llama_server_binary", None),
        "chat_format": getattr(args, "chat_format", None),
        "n_gpu_layers": getattr(args, "n_gpu_layers", None),
    }
    active: dict[str, Any] = {}
    defaults = {
        "backend": "auto",
        "n_gpu_layers": 0,
    }
    for key, value in tracked_flags.items():
        if value is None:
            continue
        if key in defaults and value == defaults[key]:
            continue
        active[key] = value
    return active
