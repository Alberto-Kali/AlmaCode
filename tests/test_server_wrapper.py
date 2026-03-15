import json
import os
import subprocess
from pathlib import Path

from almacode_server.config import ServerConfig, apply_override, default_layout, save_server_config
from almacode_server.manager import ServerManager


def _write_model(path: Path) -> None:
    path.write_text("stub", encoding="utf-8")


def test_apply_override_updates_values() -> None:
    config = ServerConfig()
    apply_override(config, "server.port=9090")
    apply_override(config, "multi_gpu.row_split=true")
    apply_override(config, "advanced.extra_flags=--cont-batching,--metrics")
    assert config.server.port == 9090
    assert config.multi_gpu.row_split is True
    assert config.advanced.extra_flags == ["--cont-batching", "--metrics"]


def test_render_command_includes_multimodal_and_multi_gpu(tmp_path: Path) -> None:
    layout = default_layout(tmp_path)
    layout.binary_path.parent.mkdir(parents=True, exist_ok=True)
    layout.binary_path.write_text("", encoding="utf-8")
    model = tmp_path / "model-vl.gguf"
    mmproj = tmp_path / "mmproj.gguf"
    _write_model(model)
    _write_model(mmproj)
    config = ServerConfig()
    config.model.model_path = str(model)
    config.model.mmproj_path = str(mmproj)
    config.multi_gpu.tensor_split = "60,40"
    config.multi_gpu.row_split = True
    config.advanced.extra_flags = ["--metrics"]
    manager = ServerManager(layout)
    command = manager.render_command(config)
    assert "--mmproj" in command
    assert "--tensor-split" in command
    assert "--split-mode" in command
    assert "--metrics" in command


def test_render_command_can_use_explicit_binary_path(tmp_path: Path) -> None:
    layout = default_layout(tmp_path)
    external_binary = tmp_path / "bin" / "llama-server"
    external_binary.parent.mkdir(parents=True, exist_ok=True)
    external_binary.write_text("", encoding="utf-8")
    model = tmp_path / "model.gguf"
    _write_model(model)

    config = ServerConfig()
    config.model.model_path = str(model)
    config.advanced.binary_path = str(external_binary)

    command = ServerManager(layout).render_command(config)
    assert command[0] == str(external_binary)


def test_validate_config_requires_mmproj_for_vision_model(tmp_path: Path) -> None:
    layout = default_layout(tmp_path)
    manager = ServerManager(layout)
    model = tmp_path / "model-vl.gguf"
    _write_model(model)
    config = ServerConfig()
    config.model.model_path = str(model)
    try:
        manager.validate_config(config)
    except Exception as exc:  # noqa: BLE001
        assert "mmproj" in str(exc)
    else:
        raise AssertionError("Expected validate_config to fail without mmproj")


def test_status_and_stop_with_pid_file(tmp_path: Path) -> None:
    layout = default_layout(tmp_path)
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(["sleep", "30"])
    try:
        layout.pid_path.write_text(str(process.pid), encoding="utf-8")
        manager = ServerManager(layout)
        assert "running" in manager.status()
        assert "Stopped" in manager.stop()
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()


def test_load_and_save_server_config(tmp_path: Path) -> None:
    layout = default_layout(tmp_path)
    config = ServerConfig()
    model = tmp_path / "model.gguf"
    _write_model(model)
    config.model.model_path = str(model)
    save_server_config(layout.config_path, config)
    loaded = ServerManager(layout).load_config()
    assert loaded.model.model_path == str(model)


def test_parse_cuda_release() -> None:
    output = "Cuda compilation tools, release 13.1, V13.1.115"
    assert ServerManager._parse_cuda_release(output) == (13, 1)
