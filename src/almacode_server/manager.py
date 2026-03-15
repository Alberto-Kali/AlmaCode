from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
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
        binary_path = self.resolve_binary_path(config)
        command = [
            str(binary_path),
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
            "--no-mmap",
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
        self.resolve_binary_path(config)

    def resolve_binary_path(self, config: ServerConfig | None = None) -> Path:
        configured = None
        if config is not None and config.advanced.binary_path:
            configured = Path(config.advanced.binary_path).expanduser().resolve()
            if configured.exists():
                return configured
            raise ServerManagerError(f"Configured llama-server binary does not exist: {configured}")
        if self.layout.binary_path.exists():
            return self.layout.binary_path
        system_binary = shutil.which("llama-server")
        if system_binary:
            return Path(system_binary).resolve()
        hint = configured or self.layout.binary_path
        raise ServerManagerError(f"llama-server binary not found. Expected one of: {hint} or llama-server in PATH")

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
        messages.append(f"Server home: {self.layout.root}")
        messages.append(f"Config path: {self.layout.config_path}")
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
            release = self._parse_cuda_release(version)
            if release is None:
                ok = False
                messages.append("Could not determine CUDA toolkit version from nvcc.")
            elif release < (12, 8):
                ok = False
                messages.append(f"CUDA toolkit 12.8 or newer is required. Found {release[0]}.{release[1]}.")
            else:
                messages.append(f"Found CUDA toolkit {release[0]}.{release[1]}.")
        if shutil.which("nvidia-smi") is None:
            ok = False
            messages.append("Missing dependency: nvidia-smi")
        else:
            messages.append("Found nvidia-smi.")
        if self.layout.config_path.exists():
            try:
                config = self.load_config()
                self.validate_config(config)
                messages.append(f"Using llama-server binary: {self.resolve_binary_path(config)}")
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
        if self.is_running():
            raise ServerManagerError("llama-server is already running.")
        command = self.render_command(config)
        log_handle = self.layout.log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        binary_dir = str(Path(command[0]).resolve().parent)
        current_ld_library_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{binary_dir}:{current_ld_library_path}" if current_ld_library_path else binary_dir
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        time.sleep(2)
        if process.poll() is not None:
            log_handle.close()
            tail = self.logs(lines=40)
            raise ServerManagerError(
                "llama-server exited during startup.\n"
                f"Log file: {self.layout.log_path}\n"
                f"{tail or 'No logs captured.'}"
            )
        self.layout.pid_path.write_text(str(process.pid), encoding="utf-8")
        return (
            f"Started llama-server with PID {process.pid}\n"
            f"Binary: {command[0]}\n"
            f"Log: {self.layout.log_path}"
        )

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
            return f"llama-server is stopped\nConfig: {self.layout.config_path}\nLog: {self.layout.log_path}"
        if self.is_running():
            return f"llama-server is running with PID {pid}\nConfig: {self.layout.config_path}\nLog: {self.layout.log_path}"
        return f"llama-server is stopped\nConfig: {self.layout.config_path}\nLog: {self.layout.log_path}"

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

    @staticmethod
    def _parse_cuda_release(version_output: str) -> tuple[int, int] | None:
        match = re.search(r"release\s+(\d+)\.(\d+)", version_output)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))
