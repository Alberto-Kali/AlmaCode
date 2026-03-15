from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from rich.console import Console

from almacode.config import AgentConfig
from almacode.llm import ContextOverflowError, LlamaBackend, ModelLoadError
from almacode.prompts import build_system_prompt, build_tool_feedback
from almacode.session import AgentSession
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
    def __init__(
        self,
        config: AgentConfig,
        backend: LlamaBackend,
        tools: WorkspaceTools,
        console: Console,
        event_handler: Callable[[str, str], None] | None = None,
    ) -> None:
        self._config = config
        self._backend = backend
        self._tools = tools
        self._console = console
        self._event_handler = event_handler

    def run(
        self,
        task: str,
        session: AgentSession | None = None,
        image_refs: list[str] | None = None,
    ) -> str:
        active_session = session or AgentSession()
        system_prompt = build_system_prompt(
            workspace=str(self._config.workspace),
            command_timeout=self._config.command_timeout,
            system_note=self._config.system_note,
        )
        user_message = build_user_message(task, image_refs=image_refs)

        for step in range(1, self._config.max_steps + 1):
            self._compact_if_needed(active_session, system_prompt, task, user_message)
            messages = active_session.build_messages(
                system_prompt=system_prompt,
                user_message=user_message,
                history_tail_messages=self._config.history_tail_messages,
            )
            try:
                response = self._backend.complete(messages)
            except ContextOverflowError as exc:
                self._emit(
                    "status",
                    (
                        f"Context full ({exc.prompt_tokens}/{exc.context_window}). "
                        "Compressing session memory and retrying."
                    ),
                )
                self._recover_from_overflow(active_session, system_prompt, task)
                continue
            except ModelLoadError:
                raise
            if self._config.verbose:
                self._console.print(f"[dim]Raw model output[{step}]:[/dim] {response.content}")

            try:
                action = parse_action(response.content)
            except Exception as exc:  # noqa: BLE001
                active_session.append_message({"role": "assistant", "content": response.content})
                active_session.append_message(
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was invalid JSON for the required schema. "
                            f"Error: {exc}. Reply again with valid JSON only."
                        ),
                    }
                )
                continue

            if action.thought:
                self._emit("thought", f"Step {step} {action.thought}")

            if action.tool == "final_answer":
                answer = str(action.args.get("answer", "")).strip()
                if not answer:
                    raise RuntimeError("Model returned final_answer without an answer")
                active_session.append_message({"role": "assistant", "content": response.content})
                self._emit("final", answer)
                return answer

            try:
                result = self._tools.execute(action.tool, action.args)
            except ToolError as exc:
                result = f"Tool error: {exc}"
            except Exception as exc:  # noqa: BLE001
                result = f"Unexpected tool failure: {exc}"

            active_session.append_message({"role": "assistant", "content": response.content})
            active_session.append_message({"role": "user", "content": build_tool_feedback(action.tool, result)})
            active_session.record_tool(action.tool, result)
            self._emit("tool", f"{action.tool}\n{result}")

        raise RuntimeError(
            f"Step limit reached ({self._config.max_steps}) before the model produced final_answer."
        )

    def _compact_if_needed(
        self,
        session: AgentSession,
        system_prompt: str,
        task: str,
        user_message: dict[str, Any],
    ) -> None:
        estimated_tokens = session.token_estimate(system_prompt, user_message)
        soft_limit = int(self._config.context_window * self._config.context_soft_limit_ratio)
        if estimated_tokens < soft_limit:
            return
        self._emit("status", f"Context nearing limit ({estimated_tokens}/{self._config.context_window}). Compacting.")
        session.compact(
            backend=self._backend,
            workspace=str(self._config.workspace),
            system_prompt=system_prompt,
            current_task=task,
            summary_max_tokens=self._config.summary_max_tokens,
            history_tail_messages=max(2, self._config.history_tail_messages),
        )
        self._emit("context", session.summary or "[empty summary]")

    def _recover_from_overflow(self, session: AgentSession, system_prompt: str, task: str) -> None:
        try:
            session.compact(
                backend=self._backend,
                workspace=str(self._config.workspace),
                system_prompt=system_prompt,
                current_task=task,
                summary_max_tokens=self._config.summary_max_tokens,
                history_tail_messages=max(2, self._config.history_tail_messages),
            )
        except ModelLoadError:
            session.hard_reset(current_task=task, history_tail_messages=2)
            self._emit("status", "Summary compaction failed. Falling back to a minimal session memory.")
        self._emit("context", session.summary or "[empty summary]")

    def _emit(self, kind: str, message: str) -> None:
        if kind == "thought":
            self._console.print(f"[cyan]{message}[/cyan]")
        elif kind == "status":
            self._console.print(f"[magenta]{message}[/magenta]")
        if self._event_handler is not None:
            self._event_handler(kind, message)
