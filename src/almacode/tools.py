from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib import parse, request


class ToolError(RuntimeError):
    pass


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
        elif tag in {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "li", "br"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag in {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "li"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        cleaned = unescape("".join(self._parts))
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip()


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
            "web_search": self.web_search,
            "open_url": self.open_url,
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
        env = self._build_command_env(resolved_cwd)
        command_timeout = timeout or self.default_timeout

        try:
            if os.name == "nt":
                completed = subprocess.run(
                    command,
                    cwd=resolved_cwd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=command_timeout,
                    env=env,
                )
            else:
                shell_executable = os.environ.get("SHELL", "/bin/bash")
                completed = subprocess.run(
                    [shell_executable, "-lc", command],
                    cwd=resolved_cwd,
                    capture_output=True,
                    text=True,
                    timeout=command_timeout,
                    env=env,
                )
        except subprocess.TimeoutExpired as exc:
            raise ToolError(
                f"Command timed out after {command_timeout} seconds: {command}"
            ) from exc
        return {
            "command": command,
            "cwd": str(resolved_cwd.relative_to(self.root) if resolved_cwd != self.root else "."),
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-12000:],
            "stderr": completed.stderr[-12000:],
            "shell": shell_executable or "system-default",
            "virtual_env": env.get("VIRTUAL_ENV"),
        }

    def _build_command_env(self, cwd: Path) -> dict[str, str]:
        env = os.environ.copy()
        venv_path = self._find_preferred_venv(cwd)
        if venv_path is None:
            return env

        bin_dir = venv_path / ("Scripts" if os.name == "nt" else "bin")
        env["VIRTUAL_ENV"] = str(venv_path)
        path_parts = [str(bin_dir)]
        existing_path = env.get("PATH", "")
        if existing_path:
            path_parts.append(existing_path)
        env["PATH"] = os.pathsep.join(path_parts)
        return env

    def _find_preferred_venv(self, cwd: Path) -> Path | None:
        for parent in [cwd, *cwd.parents]:
            if parent == self.root.parent:
                break
            for name in (".venv", "venv", "env"):
                candidate = parent / name
                if self._is_virtualenv(candidate):
                    return candidate
            child_candidates = [
                child
                for child in sorted(parent.iterdir(), key=lambda item: item.name.lower())
                if child.is_dir() and self._is_virtualenv(child)
            ]
            if len(child_candidates) == 1:
                return child_candidates[0]
            if parent == self.root:
                break
        return None

    @staticmethod
    def _is_virtualenv(path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        if not (path / "pyvenv.cfg").exists():
            return False
        bin_dir = path / ("Scripts" if os.name == "nt" else "bin")
        python_name = "python.exe" if os.name == "nt" else "python"
        return (bin_dir / python_name).exists()

    def web_search(self, query: str, limit: int = 5) -> dict[str, Any]:
        if not query.strip():
            raise ToolError("query must not be empty")
        if limit < 1 or limit > 10:
            raise ToolError("limit must be between 1 and 10")

        url = "https://html.duckduckgo.com/html/?" + parse.urlencode({"q": query})
        req = request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 AlmaCode/1.0",
                "Accept-Language": "en-US,en;q=0.8",
            },
        )
        try:
            with request.urlopen(req, timeout=20) as response:
                html = response.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"Web search failed: {exc}") from exc

        matches = re.findall(
            r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            html,
            flags=re.IGNORECASE | re.DOTALL,
        )
        results: list[dict[str, str]] = []
        for href, raw_title in matches:
            title = re.sub(r"<.*?>", "", raw_title)
            title = unescape(title).strip()
            parsed = parse.urlparse(href)
            if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
                target = parse.parse_qs(parsed.query).get("uddg", [href])[0]
            else:
                target = href
            results.append({"title": title, "url": target})
            if len(results) >= limit:
                break
        if not results:
            raise ToolError("Web search returned no results")
        return {"query": query, "results": results}

    def open_url(self, url: str, max_chars: int = 12000) -> dict[str, Any]:
        parsed = parse.urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ToolError("url must start with http:// or https://")
        req = request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 AlmaCode/1.0",
                "Accept-Language": "en-US,en;q=0.8",
            },
        )
        try:
            with request.urlopen(req, timeout=20) as response:
                content_type = response.headers.get("Content-Type", "")
                body = response.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"Failed to open URL: {exc}") from exc

        extractor = _HTMLTextExtractor()
        extractor.feed(body)
        text = extractor.text()
        if not text and "html" not in content_type.lower():
            text = body
        return {
            "url": url,
            "content_type": content_type,
            "text": text[:max_chars],
        }
