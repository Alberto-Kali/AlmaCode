from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from almacode.config import AgentConfig
from almacode.llm import LlamaBackend
from almacode.prompts import build_system_prompt, build_tool_feedback
from almacode.tools import ToolError, WorkspaceTools


@dataclass(slots=True)
class AgentAction:
    tool: str
    args: dict[str, Any]
    thought: str = ""


def _image_ref_to_url(image_ref: str) -> str:
    if image_ref.startswith(("http://", "https://", "data:")):
        return image_ref

    image_path = Path(image_ref).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image file does not exist: {image_ref}")
    if not image_path.is_file():
        raise ValueError(f"Image path is not a file: {image_ref}")

    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type is None:
        mime_type = "application/octet-stream"

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_user_message(task: str, image_refs: list[str] | None = None) -> dict[str, Any]:
    if not image_refs:
        return {"role": "user", "content": task}

    content: list[dict[str, Any]] = [{"type": "text", "text": task}]
    for image_ref in image_refs:
        content.append({"type": "image_url", "image_url": {"url": _image_ref_to_url(image_ref)}})
    return {"role": "user", "content": content}


def _extract_first_json_block(raw: str) -> str:
    start = raw.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return raw[start : index + 1]

    raise ValueError("Unterminated JSON object in model output")


def parse_action(raw: str) -> AgentAction:
    payload = json.loads(_extract_first_json_block(raw))

    if "action" in payload and isinstance(payload["action"], dict):
        action = payload["action"]
        thought = str(payload.get("thought", ""))
    else:
        action = payload
        thought = str(payload.get("thought", ""))

    tool = action.get("tool")
    if not isinstance(tool, str) or not tool:
        raise ValueError("Missing action.tool in model output")

    args = action.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("action.args must be an object")

    if tool == "final_answer" and "answer" not in args:
        if "final_answer" in payload:
            args["answer"] = payload["final_answer"]
        elif "answer" in payload:
            args["answer"] = payload["answer"]

    return AgentAction(tool=tool, args=args, thought=thought)


class CodingAgent:
    def __init__(self, config: AgentConfig, backend: LlamaBackend, tools: WorkspaceTools, console: Console) -> None:
        self._config = config
        self._backend = backend
        self._tools = tools
        self._console = console

    def run(self, task: str, history: list[dict[str, Any]] | None = None, image_refs: list[str] | None = None) -> str:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": build_system_prompt(
                    workspace=str(self._config.workspace),
                    command_timeout=self._config.command_timeout,
                    system_note=self._config.system_note,
                ),
            }
        ]
        if history:
            messages.extend(history)
        messages.append(build_user_message(task, image_refs=image_refs))

        for step in range(1, self._config.max_steps + 1):
            response = self._backend.complete(messages)
            if self._config.verbose:
                self._console.print(f"[dim]Raw model output[{step}]:[/dim] {response.content}")

            try:
                action = parse_action(response.content)
            except Exception as exc:  # noqa: BLE001
                messages.append({"role": "assistant", "content": response.content})
                messages.append(
                    {
                        "role": "user",
                        "content": f"Your previous reply was invalid JSON for the required schema. Error: {exc}. Reply again with valid JSON only.",
                    }
                )
                continue

            if action.thought:
                self._console.print(f"[cyan]Step {step}[/cyan] {action.thought}")

            if action.tool == "final_answer":
                answer = str(action.args.get("answer", "")).strip()
                if not answer:
                    raise RuntimeError("Model returned final_answer without an answer")
                return answer

            try:
                result = self._tools.execute(action.tool, action.args)
            except ToolError as exc:
                result = f"Tool error: {exc}"
            except Exception as exc:  # noqa: BLE001
                result = f"Unexpected tool failure: {exc}"

            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": build_tool_feedback(action.tool, result)})

        raise RuntimeError(
            f"Step limit reached ({self._config.max_steps}) before the model produced final_answer."
        )
