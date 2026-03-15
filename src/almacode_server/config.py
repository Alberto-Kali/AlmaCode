from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
DEFAULT_CTX = 8192
DEFAULT_GPU_LAYERS = -1
LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_REF = "b5092"


@dataclass(slots=True)
class ServerSection:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT


@dataclass(slots=True)
class ModelSection:
    model_path: str = ""
    mmproj_path: str | None = None
    chat_template: str | None = None


@dataclass(slots=True)
class RuntimeSection:
    ctx_size: int = DEFAULT_CTX
    gpu_layers: int = DEFAULT_GPU_LAYERS
    threads: int | None = None
    threads_batch: int | None = None
    batch_size: int = 1024
    ubatch_size: int = 1024
    flash_attn: bool = True


@dataclass(slots=True)
class MultiGPUSection:
    tensor_split: str | None = None
    row_split: bool = False


@dataclass(slots=True)
class AdvancedSection:
    binary_path: str | None = None
    extra_flags: list[str] = field(default_factory=list)
    cache_type_k: str | None = None
    cache_type_v: str | None = None
    numa: bool = False
    no_mmap: bool = False
    mlock: bool = False


@dataclass(slots=True)
class ServerConfig:
    server: ServerSection = field(default_factory=ServerSection)
    model: ModelSection = field(default_factory=ModelSection)
    runtime: RuntimeSection = field(default_factory=RuntimeSection)
    multi_gpu: MultiGPUSection = field(default_factory=MultiGPUSection)
    advanced: AdvancedSection = field(default_factory=AdvancedSection)


@dataclass(slots=True)
class RuntimeLayout:
    root: Path
    config_dir: Path
    logs_dir: Path
    run_dir: Path
    runtime_dir: Path
    src_dir: Path
    build_dir: Path
    binary_path: Path
    config_path: Path
    log_path: Path
    pid_path: Path


def default_layout(root: Path | None = None) -> RuntimeLayout:
    base = Path(root or os.environ.get("ALMACODE_SERVER_HOME") or Path.home() / ".local" / "share" / "almacode-server").expanduser().resolve()
    config_dir = base / "config"
    logs_dir = base / "logs"
    run_dir = base / "run"
    runtime_dir = base / "runtime"
    src_dir = runtime_dir / "src"
    build_dir = runtime_dir / "build"
    binary_path = build_dir / "bin" / "llama-server"
    config_path = config_dir / "server.json"
    log_path = logs_dir / "server.log"
    pid_path = run_dir / "llama-server.pid"
    return RuntimeLayout(
        root=base,
        config_dir=config_dir,
        logs_dir=logs_dir,
        run_dir=run_dir,
        runtime_dir=runtime_dir,
        src_dir=src_dir,
        build_dir=build_dir,
        binary_path=binary_path,
        config_path=config_path,
        log_path=log_path,
        pid_path=pid_path,
    )


def ensure_layout(layout: RuntimeLayout) -> None:
    for path in (layout.root, layout.config_dir, layout.logs_dir, layout.run_dir, layout.runtime_dir, layout.src_dir):
        path.mkdir(parents=True, exist_ok=True)


def to_dict(config: ServerConfig) -> dict[str, Any]:
    return asdict(config)


def load_server_config(path: Path) -> ServerConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ServerConfig(
        server=ServerSection(**payload.get("server", {})),
        model=ModelSection(**payload.get("model", {})),
        runtime=RuntimeSection(**payload.get("runtime", {})),
        multi_gpu=MultiGPUSection(**payload.get("multi_gpu", {})),
        advanced=AdvancedSection(**payload.get("advanced", {})),
    )


def save_server_config(path: Path, config: ServerConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(config), ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def apply_override(config: ServerConfig, assignment: str) -> None:
    if "=" not in assignment or "." not in assignment:
        raise ValueError(f"Invalid override: {assignment}. Use section.key=value")
    dotted_key, raw_value = assignment.split("=", 1)
    section_name, field_name = dotted_key.split(".", 1)
    section = getattr(config, section_name, None)
    if section is None or not hasattr(section, field_name):
        raise ValueError(f"Unknown config key: {dotted_key}")
    current_value = getattr(section, field_name)
    if isinstance(current_value, bool):
        parsed: Any = raw_value.lower() in {"1", "true", "yes", "on"}
    elif isinstance(current_value, int):
        parsed = int(raw_value)
    elif isinstance(current_value, list):
        parsed = [item.strip() for item in raw_value.split(",") if item.strip()]
    elif raw_value.lower() in {"none", "null", ""}:
        parsed = None
    else:
        parsed = raw_value
    setattr(section, field_name, parsed)
