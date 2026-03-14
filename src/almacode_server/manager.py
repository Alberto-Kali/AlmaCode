from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

from almacode_server.config import (
    LLAMA_CPP_REF,
    LLAMA_CPP_REPO,
    RuntimeLayout,
    ServerConfig,
    default_layout,
    ensure_layout,
    load_server_config,
)


class ServerManagerError(RuntimeError):
    pass


@dataclass(slots=True)
class DoctorReport:
    ok: bool
    messages: list[str]


class ServerManager:
    def __init__(self, layout: RuntimeLayout | None = None) -> None:
        self.layout = layout or default_layout()

    def load_config(self) -> ServerConfig:
        if not self.layout.config_path.exists():
            raise ServerManagerError(f"Missing config file: {self.layout.config_path}")
        return load_server_config(self.layout.config_path)

    def render_command(self, config: ServerConfig) -> list[str]:
        command = [
            str(self.layout.binary_path),
            "--model",
            config.model.model_path,
            "--host",
            config.server.host,
            "--port",
            str(config.server.port),
            "--ctx-size",
            str(config.runtime.ctx_size),
            "--gpu-layers",
            str(config.runtime.gpu_layers),
            "--batch-size",
            str(config.runtime.batch_size),
            "--ubatch-size",
            str(config.runtime.ubatch_size),
            "--jinja",
        ]
        if config.runtime.flash_attn:
            command.extend(["--flash-attn", "on"])
        if config.model.mmproj_path:
            command.extend(["--mmproj", config.model.mmproj_path])
        if config.model.chat_template:
            command.extend(["--chat-template", config.model.chat_template])
        if config.runtime.threads is not None:
            command.extend(["--threads", str(config.runtime.threads)])
        if config.runtime.threads_batch is not None:
            command.extend(["--threads-batch", str(config.runtime.threads_batch)])
        if config.multi_gpu.tensor_split:
            command.extend(["--tensor-split", config.multi_gpu.tensor_split])
        if config.multi_gpu.row_split:
            command.extend(["--split-mode", "row"])
        if config.advanced.cache_type_k and config.advanced.cache_type_v:
            command.extend(["--cache-type-k", config.advanced.cache_type_k, "--cache-type-v", config.advanced.cache_type_v])
        if config.advanced.numa:
            command.extend(["--numa", "distribute"])
        if config.advanced.no_mmap:
            command.append("--no-mmap")
        if config.advanced.mlock:
            command.append("--mlock")
        command.extend(config.advanced.extra_flags)
        return command

    def validate_config(self, config: ServerConfig) -> None:
        if not config.model.model_path:
            raise ServerManagerError("config.model.model_path must not be empty")
        model_path = Path(config.model.model_path).expanduser()
        if not model_path.exists():
            raise ServerManagerError(f"Model file does not exist: {model_path}")
        lower_name = model_path.name.lower()
        if any(token in lower_name for token in ("-vl", "vl-", "vision")) and not config.model.mmproj_path:
            raise ServerManagerError("Vision models require config.model.mmproj_path")
        if config.model.mmproj_path and not Path(config.model.mmproj_path).expanduser().exists():
            raise ServerManagerError(f"mmproj file does not exist: {config.model.mmproj_path}")
        if config.multi_gpu.tensor_split and not re.fullmatch(r"\d+(\.\d+)?(,\d+(\.\d+)?)+", config.multi_gpu.tensor_split):
            raise ServerManagerError("multi_gpu.tensor_split must be a comma-separated list like 60,40")

    def install_runtime(self) -> list[str]:
        ensure_layout(self.layout)
        source_dir = self.layout.src_dir / "llama.cpp"
        if not source_dir.exists():
            self._run(["git", "clone", LLAMA_CPP_REPO, str(source_dir)])
        self._run(["git", "-C", str(source_dir), "fetch", "--tags", "--force"])
        self._run(["git", "-C", str(source_dir), "checkout", LLAMA_CPP_REF])
        self.layout.build_dir.mkdir(parents=True, exist_ok=True)
        configure = [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(self.layout.build_dir),
            "-DGGML_CUDA=ON",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        self._run(configure)
        if shutil.which("ninja"):
            self._run(["cmake", "--build", str(self.layout.build_dir), "--target", "llama-server", "-j"])
        else:
            self._run(["cmake", "--build", str(self.layout.build_dir), "--target", "llama-server", "-j"])
        return [
            f"Checked out llama.cpp {LLAMA_CPP_REF}",
            f"Built llama-server at {self.layout.binary_path}",
        ]

    def doctor(self) -> DoctorReport:
        messages: list[str] = []
        ok = True
        if os.uname().sysname != "Linux":
            ok = False
            messages.append("Linux is required for the server package.")
        if os.uname().machine not in {"x86_64", "amd64"}:
            ok = False
            messages.append("x86_64 is required for the server package.")
        for command in ("git", "cmake"):
            if shutil.which(command) is None:
                ok = False
                messages.append(f"Missing dependency: {command}")
        if shutil.which("ninja") is None and shutil.which("make") is None:
            ok = False
            messages.append("Missing dependency: ninja or make")
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            ok = False
            messages.append("Missing dependency: nvcc (CUDA toolkit 12.8)")
        else:
            version = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=False).stdout
            if "release 12.8" not in version:
                ok = False
                messages.append("CUDA toolkit 12.8 is required.")
        if shutil.which("nvidia-smi") is None:
            ok = False
            messages.append("Missing dependency: nvidia-smi")
        if self.layout.config_path.exists():
            try:
                config = self.load_config()
                self.validate_config(config)
            except ServerManagerError as exc:
                ok = False
                messages.append(str(exc))
        else:
            messages.append(f"Config not found yet: {self.layout.config_path}")
        if not messages:
            messages.append("Environment looks good.")
        return DoctorReport(ok=ok, messages=messages)

    def start(self) -> str:
        ensure_layout(self.layout)
        config = self.load_config()
        self.validate_config(config)
        if not self.layout.binary_path.exists():
            raise ServerManagerError(f"llama-server binary not found: {self.layout.binary_path}")
        if self.is_running():
            raise ServerManagerError("llama-server is already running.")
        command = self.render_command(config)
        log_handle = self.layout.log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True)
        self.layout.pid_path.write_text(str(process.pid), encoding="utf-8")
        return f"Started llama-server with PID {process.pid}"

    def stop(self) -> str:
        pid = self._read_pid()
        if pid is None:
            return "llama-server is not running."
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if self.layout.pid_path.exists():
            self.layout.pid_path.unlink()
        return f"Stopped llama-server PID {pid}"

    def is_running(self) -> bool:
        pid = self._read_pid()
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    def status(self) -> str:
        pid = self._read_pid()
        if pid is None:
            return "llama-server is stopped"
        return f"llama-server is running with PID {pid}" if self.is_running() else "llama-server is stopped"

    def logs(self, lines: int = 50) -> str:
        if not self.layout.log_path.exists():
            return "No logs yet."
        content = self.layout.log_path.read_text(encoding="utf-8")
        return "\n".join(content.splitlines()[-lines:])

    def _read_pid(self) -> int | None:
        if not self.layout.pid_path.exists():
            return None
        try:
            return int(self.layout.pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            return None

    @staticmethod
    def _run(command: list[str]) -> None:
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise ServerManagerError(f"Command failed with exit code {completed.returncode}: {' '.join(command)}")
