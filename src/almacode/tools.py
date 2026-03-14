from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ToolError(RuntimeError):
    pass


@dataclass(slots=True)
class WorkspaceTools:
    root: Path
    default_timeout: int

    def __post_init__(self) -> None:
        self.root = self.root.expanduser().resolve()

    def execute(self, tool: str, args: dict[str, Any]) -> str:
        dispatch = {
            "list_dir": self.list_dir,
            "read_file": self.read_file,
            "write_file": self.write_file,
            "replace_in_file": self.replace_in_file,
            "make_dir": self.make_dir,
            "run_command": self.run_command,
        }
        if tool not in dispatch:
            raise ToolError(f"Unknown tool: {tool}")
        result = dispatch[tool](**args)
        if isinstance(result, str):
            return result
        return json.dumps(result, ensure_ascii=True, indent=2)

    def resolve_path(self, raw_path: str) -> Path:
        candidate = (self.root / raw_path).expanduser().resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ToolError(f"Path escapes the workspace: {raw_path}")
        return candidate

    def list_dir(self, path: str = ".") -> dict[str, Any]:
        target = self.resolve_path(path)
        if not target.exists():
            raise ToolError(f"Directory does not exist: {path}")
        if not target.is_dir():
            raise ToolError(f"Path is not a directory: {path}")

        entries: list[dict[str, Any]] = []
        for child in sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            entries.append(
                {
                    "name": child.name,
                    "type": "dir" if child.is_dir() else "file",
                    "size": child.stat().st_size if child.is_file() else None,
                }
            )
        return {"path": str(target.relative_to(self.root) if target != self.root else "."), "entries": entries}

    def read_file(self, path: str, start_line: int = 1, max_lines: int = 200) -> str:
        target = self.resolve_path(path)
        if not target.exists():
            raise ToolError(f"File does not exist: {path}")
        if not target.is_file():
            raise ToolError(f"Path is not a file: {path}")
        if start_line < 1:
            raise ToolError("start_line must be >= 1")
        if max_lines < 1:
            raise ToolError("max_lines must be >= 1")

        lines = target.read_text(encoding="utf-8").splitlines()
        start_index = start_line - 1
        selected = lines[start_index : start_index + max_lines]
        rendered = [f"{index}: {line}" for index, line in enumerate(selected, start=start_line)]
        return "\n".join(rendered) if rendered else "[empty selection]"

    def write_file(self, path: str, content: str, append: bool = False) -> dict[str, Any]:
        target = self.resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with target.open(mode, encoding="utf-8") as handle:
            handle.write(content)
        return {"path": path, "bytes_written": len(content.encode("utf-8")), "append": append}

    def replace_in_file(self, path: str, old: str, new: str, count: int = 0) -> dict[str, Any]:
        if not old:
            raise ToolError("old must not be empty")
        target = self.resolve_path(path)
        if not target.exists():
            raise ToolError(f"File does not exist: {path}")
        original = target.read_text(encoding="utf-8")
        replacements = original.count(old) if count == 0 else min(original.count(old), count)
        if replacements == 0:
            raise ToolError(f"Text not found in {path}")
        updated = original.replace(old, new) if count == 0 else original.replace(old, new, count)
        target.write_text(updated, encoding="utf-8")
        return {"path": path, "replacements": replacements}

    def make_dir(self, path: str) -> dict[str, Any]:
        target = self.resolve_path(path)
        target.mkdir(parents=True, exist_ok=True)
        return {"path": path, "created": True}

    def run_command(self, command: str, cwd: str = ".", timeout: int | None = None) -> dict[str, Any]:
        if not command.strip():
            raise ToolError("command must not be empty")

        resolved_cwd = self.resolve_path(cwd)
        if not resolved_cwd.is_dir():
            raise ToolError(f"cwd is not a directory: {cwd}")

        shell_executable = None
        if os.name != "nt":
            shell_executable = os.environ.get("SHELL", "/bin/bash")

        try:
            completed = subprocess.run(
                command,
                cwd=resolved_cwd,
                shell=True,
                executable=shell_executable,
                capture_output=True,
                text=True,
                timeout=timeout or self.default_timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(
                f"Command timed out after {timeout or self.default_timeout} seconds: {command}"
            ) from exc
        return {
            "command": command,
            "cwd": str(resolved_cwd.relative_to(self.root) if resolved_cwd != self.root else "."),
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-12000:],
            "stderr": completed.stderr[-12000:],
        }
