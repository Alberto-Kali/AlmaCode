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
from almacode.prompts import (
    build_command_output_summary_prompt,
    build_plan_prompt,
    build_research_summary_prompt,
    build_step_prompt,
    build_system_prompt,
    build_tool_feedback,
)
from almacode.session import AgentSession, PlanStep
from almacode.tools import ToolError, WorkspaceTools


@dataclass(slots=True)
class AgentAction:
    tool: str
    args: dict[str, Any]
    thought: str = ""


class AgentRuntimeError(RuntimeError):
    pass


class StepLimitReachedError(AgentRuntimeError):
    pass


COMMAND_OUTPUT_WINDOW_CHARS = 4000
COMMAND_OUTPUT_FEEDBACK_CHARS = 800


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
        active_session.last_user_task = task
        system_prompt = build_system_prompt(
            workspace=str(self._config.workspace),
            command_timeout=self._config.command_timeout,
            system_note=self._config.system_note,
        )
        root_user_message = build_user_message(task, image_refs=image_refs)
        active_session.append_message(root_user_message)
        if active_session.plan_task != task or not active_session.plan_steps:
            self._prepare_plan(active_session, task)
        self._emit("plan", active_session.render_plan())

        repeated_action_count = 0
        last_action_fingerprint: tuple[str, str, str] | None = None
        substep = 0

        while True:
            current_step = active_session.current_step()
            if current_step is None:
                answer = self._finalize_completed_plan(active_session, task)
                self._emit("final", answer)
                return answer
            if substep >= self._config.max_steps:
                raise StepLimitReachedError(
                    "Substep limit reached "
                    f"({self._config.max_steps}) for current plan step: {current_step.title}"
                )
            substep += 1
            step_prompt = build_step_prompt(
                task=task,
                step_index=active_session.active_step_index + 1,
                total_steps=len(active_session.plan_steps),
                step_title=current_step.title,
                step_details=current_step.details,
                completed_steps=self._completed_step_lines(active_session),
                research_summary=active_session.research_summary,
            )
            user_message = build_user_message(step_prompt, image_refs=image_refs)
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
                self._recover_from_overflow(active_session, system_prompt, task, user_message)
                continue
            except ModelLoadError:
                raise
            if self._config.verbose:
                self._console.print(f"[dim]Raw model output[{substep}]:[/dim] {response.content}")

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
                self._emit(
                    "thought",
                    f"Plan {active_session.active_step_index + 1}/{len(active_session.plan_steps)} "
                    f"substep {substep} {action.thought}",
                )

            action_fingerprint = (
                action.tool,
                json.dumps(action.args, ensure_ascii=True, sort_keys=True),
                action.thought.strip(),
            )
            if action_fingerprint == last_action_fingerprint:
                repeated_action_count += 1
            else:
                repeated_action_count = 0
                last_action_fingerprint = action_fingerprint

            if repeated_action_count >= 2:
                self._handle_repeated_loop(
                    session=active_session,
                    response_content=response.content,
                    system_prompt=system_prompt,
                    task=task,
                    user_message=user_message,
                    loop_tool=action.tool,
                    loop_thought=action.thought,
                )
                repeated_action_count = 0
                last_action_fingerprint = None
                continue

            if action.tool == "complete_step":
                step_summary = str(action.args.get("summary", "")).strip() or "Step completed."
                active_session.append_message({"role": "assistant", "content": response.content})
                active_session.complete_current_step(step_summary)
                active_session.compact_completed_step(
                    task=task,
                    history_tail_messages=max(2, self._config.history_tail_messages // 2),
                )
                self._emit("status", f"Completed plan step: {current_step.title}")
                self._emit("context", active_session.summary or "[empty summary]")
                self._emit("plan", active_session.render_plan())
                repeated_action_count = 0
                last_action_fingerprint = None
                substep = 0
                continue

            if action.tool == "final_answer":
                answer = str(action.args.get("answer", "")).strip()
                if not answer:
                    raise RuntimeError("Model returned final_answer without an answer")
                active_session.append_message({"role": "assistant", "content": response.content})
                self._emit("final", answer)
                return answer

            self._emit("tool_start", self._format_tool_start(action.tool, action.args))
            try:
                result = self._tools.execute(action.tool, action.args)
            except ToolError as exc:
                result = f"Tool error: {exc}"
            except Exception as exc:  # noqa: BLE001
                result = f"Unexpected tool failure: {exc}"
            research_note = self._research_from_failure(task, current_step, action.tool, action.args, result)
            if research_note:
                result = f"{result}\n\nResearch note:\n{research_note}"
            result = self._prepare_result_for_context(action.tool, result)

            active_session.append_message({"role": "assistant", "content": response.content})
            active_session.append_message({"role": "user", "content": build_tool_feedback(action.tool, result)})
            active_session.record_tool(action.tool, result)
            self._emit("tool", self._format_tool_event(action.tool, action.args, result))

    def _prepare_plan(self, session: AgentSession, task: str) -> None:
        research_summary = self._research_task_background(task)
        plan_prompt = build_plan_prompt(task, str(self._config.workspace), research_summary)
        normalized_steps = self._request_plan_steps(plan_prompt, task)
        if not normalized_steps:
            raise AgentRuntimeError("Planner returned empty step titles.")
        session.set_plan(task=task, steps=normalized_steps, research_summary=research_summary)
        self._emit("status", f"Prepared a {len(normalized_steps)}-step plan.")

    def _request_plan_steps(self, plan_prompt: str, task: str) -> list[dict[str, str]]:
        if not hasattr(self._backend, "chat"):
            return [{"title": task[:80], "details": "Complete the requested task directly."}]
        response = self._backend.chat(
            [
                {"role": "system", "content": "Return only valid JSON with a short plan."},
                {"role": "user", "content": plan_prompt},
            ],
            response_format={"type": "json_object"},
        )
        try:
            payload = json.loads(response.content)
        except json.JSONDecodeError as exc:
            raise AgentRuntimeError(f"Planner returned invalid JSON: {response.content}") from exc
        steps = payload.get("steps", [])
        if not isinstance(steps, list) or not steps:
            return [{"title": task[:80], "details": "Complete the requested task directly."}]
        normalized_steps: list[dict[str, str]] = []
        for step in steps[: self._config.max_plan_steps]:
            if not isinstance(step, dict):
                continue
            title = str(step.get("title", "")).strip()
            if not title:
                continue
            normalized_steps.append({"title": title, "details": str(step.get("details", "")).strip()})
        return normalized_steps

    def _research_task_background(self, task: str) -> str:
        queries = self._build_research_queries(task)
        if not queries:
            return ""
        notes: list[str] = []
        for query in queries[:2]:
            try:
                search_payload = json.loads(self._tools.execute("web_search", {"query": query, "limit": 3}))
            except Exception:  # noqa: BLE001
                continue
            results = search_payload.get("results", [])
            if not isinstance(results, list) or not results:
                continue
            notes.append(f"Query: {query}")
            for result in results[:2]:
                if not isinstance(result, dict):
                    continue
                title = str(result.get("title", "")).strip()
                url = str(result.get("url", "")).strip()
                if not url:
                    continue
                notes.append(f"- {title} ({url})")
                try:
                    opened = json.loads(self._tools.execute("open_url", {"url": url, "max_chars": 4000}))
                except Exception:  # noqa: BLE001
                    continue
                notes.append(str(opened.get("text", ""))[:1200])
        if not notes:
            return ""
        prompt = build_research_summary_prompt(task, "\n".join(notes))
        summary = self._backend.summarize(
            [
                {"role": "system", "content": "Return only a concise research summary."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=min(220, self._config.summary_max_tokens),
        )
        self._emit("status", "Prepared research notes for the task.")
        return summary.strip()

    @staticmethod
    def _build_research_queries(task: str) -> list[str]:
        lowered = task.lower()
        need_research = any(
            token in lowered
            for token in [
                "error",
                "traceback",
                "exception",
                "tutorial",
                "install",
                "setup",
                "fastapi",
                "django",
                "react",
                "nextjs",
                "cuda",
                "llama",
            ]
        )
        if not need_research:
            return []
        return [task, f"{task} tutorial", f"{task} fix"][:3]

    def _research_from_failure(
        self,
        task: str,
        current_step: PlanStep,
        tool_name: str,
        args: dict[str, Any],
        result: str,
    ) -> str:
        if tool_name != "run_command":
            return ""
        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            return ""
        if payload.get("exit_code", 0) == 0:
            return ""
        stderr = str(payload.get("stderr", "")).strip()
        stdout = str(payload.get("stdout", "")).strip()
        error_snippet = stderr or stdout
        if not error_snippet:
            return ""
        query = f"{args.get('command', '')} {error_snippet[:220]}"
        try:
            search_payload = json.loads(self._tools.execute("web_search", {"query": query, "limit": 3}))
        except Exception:  # noqa: BLE001
            return ""
        results = search_payload.get("results", [])
        if not isinstance(results, list) or not results:
            return ""
        note_lines = [f"Failure research for step '{current_step.title}':", f"Query: {query}"]
        for result_entry in results[:2]:
            if isinstance(result_entry, dict):
                note_lines.append(f"- {result_entry.get('title', '')}: {result_entry.get('url', '')}")
        self._emit("status", "Searched the web for the recent command failure.")
        return "\n".join(note_lines)

    @staticmethod
    def _completed_step_lines(session: AgentSession) -> list[str]:
        lines: list[str] = []
        for index, step in enumerate(session.plan_steps, start=1):
            if step.status == "completed":
                summary = f" - {step.summary}" if step.summary else ""
                lines.append(f"{index}. {step.title}{summary}")
        return lines

    def _finalize_completed_plan(self, session: AgentSession, task: str) -> str:
        lines = [f"Completed task: {task}", "", "Plan progress:"]
        for index, step in enumerate(session.plan_steps, start=1):
            summary = step.summary or step.status
            lines.append(f"{index}. {step.title} -> {summary}")
        if session.research_summary:
            lines.extend(["", "Research summary:", session.research_summary])
        return "\n".join(lines).strip()

    def _prepare_result_for_context(self, tool_name: str, result: str) -> str:
        if tool_name != "run_command":
            return result
        return self._compact_command_result(result)

    def _compact_command_result(self, result: str) -> str:
        try:
            payload = json.loads(result)
        except json.JSONDecodeError:
            return result

        stdout = str(payload.get("stdout", ""))
        stderr = str(payload.get("stderr", ""))
        rendered_output = self._render_command_output(stdout, stderr)
        if rendered_output.strip():
            output_window = rendered_output
            output_truncated = False
            if len(rendered_output) > COMMAND_OUTPUT_WINDOW_CHARS:
                output_window = rendered_output[-COMMAND_OUTPUT_WINDOW_CHARS:]
                output_truncated = True
            payload["output_summary"] = self._summarize_command_output(
                command=str(payload.get("command", "")),
                cwd=str(payload.get("cwd", ".")),
                exit_code=payload.get("exit_code", "?"),
                output_text=output_window,
                output_truncated=output_truncated,
            )
            payload["output_window_chars"] = len(output_window)
            payload["output_original_chars"] = len(rendered_output)
            payload["output_truncated"] = output_truncated

        payload["stdout"] = self._trim_command_stream(stdout)
        payload["stderr"] = self._trim_command_stream(stderr)
        return json.dumps(payload, ensure_ascii=True, indent=2)

    def _summarize_command_output(
        self,
        *,
        command: str,
        cwd: str,
        exit_code: int | str,
        output_text: str,
        output_truncated: bool,
    ) -> str:
        prompt = build_command_output_summary_prompt(
            command=command,
            cwd=cwd,
            exit_code=exit_code,
            output_text=output_text,
            output_truncated=output_truncated,
        )
        try:
            summary = self._backend.summarize(
                [
                    {"role": "system", "content": "Return only a concise command-output summary."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=min(220, self._config.summary_max_tokens),
            )
            summary = summary.strip()
            if summary:
                return summary
        except Exception:  # noqa: BLE001
            pass
        return self._heuristic_shorten(output_text, 1200)

    @staticmethod
    def _render_command_output(stdout: str, stderr: str) -> str:
        parts: list[str] = []
        if stdout.strip():
            parts.append("stdout:\n" + stdout.strip())
        if stderr.strip():
            parts.append("stderr:\n" + stderr.strip())
        return "\n\n".join(parts)

    @staticmethod
    def _trim_command_stream(text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return ""
        if len(stripped) <= COMMAND_OUTPUT_FEEDBACK_CHARS:
            return stripped
        return "[trimmed to tail]\n" + stripped[-COMMAND_OUTPUT_FEEDBACK_CHARS:]

    @staticmethod
    def _heuristic_shorten(text: str, limit: int) -> str:
        compact = " ".join(text.split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3] + "..."

    def _handle_repeated_loop(
        self,
        session: AgentSession,
        response_content: str,
        system_prompt: str,
        task: str,
        user_message: dict[str, Any],
        loop_tool: str,
        loop_thought: str,
    ) -> None:
        session.append_message({"role": "assistant", "content": response_content})
        self._emit("status", "Detected repeated step loop. Compacting context and resetting strategy.")
        loop_notice = (
            "Loop detected: you repeated the same step at least three times without meaningful progress. "
            f"Repeated tool: {loop_tool}. "
            f"Repeated thought: {loop_thought or '[empty]'}. "
            "The previous context was compacted. Continue from the summary below, do not repeat that step again, "
            "and either pick a different concrete action or return final_answer if the task is complete or blocked."
        )
        try:
            session.compact(
                backend=self._backend,
                workspace=str(self._config.workspace),
                system_prompt=system_prompt,
                current_task=task,
                summary_max_tokens=self._config.summary_max_tokens,
                history_tail_messages=max(4, self._config.history_tail_messages),
            )
        except ModelLoadError:
            session.hard_reset(current_task=task, history_tail_messages=4)
            self._emit("status", "Loop compaction failed. Falling back to a minimal session memory.")
        self._ensure_current_prompt_at_end(session, user_message)
        session.append_message({"role": "user", "content": loop_notice})
        self._emit("context", session.summary or "[empty summary]")

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
            history_tail_messages=max(4, self._config.history_tail_messages),
        )
        self._ensure_current_prompt_at_end(session, user_message)
        self._emit("context", session.summary or "[empty summary]")

    def _recover_from_overflow(
        self,
        session: AgentSession,
        system_prompt: str,
        task: str,
        user_message: dict[str, Any],
    ) -> None:
        try:
            session.compact(
                backend=self._backend,
                workspace=str(self._config.workspace),
                system_prompt=system_prompt,
                current_task=task,
                summary_max_tokens=self._config.summary_max_tokens,
                history_tail_messages=max(4, self._config.history_tail_messages),
            )
        except ModelLoadError:
            session.hard_reset(current_task=task, history_tail_messages=4)
            self._emit("status", "Summary compaction failed. Falling back to a minimal session memory.")
        self._ensure_current_prompt_at_end(session, user_message)
        self._emit("context", session.summary or "[empty summary]")

    @staticmethod
    def _ensure_current_prompt_at_end(session: AgentSession, user_message: dict[str, Any]) -> None:
        filtered_history = [message for message in session.recent_history if message != user_message]
        filtered_history.append(user_message)
        session.recent_history = filtered_history

    def _emit(self, kind: str, message: str) -> None:
        if kind == "thought":
            self._console.print(f"[cyan]{message}[/cyan]")
        elif kind == "status":
            self._console.print(f"[magenta]{message}[/magenta]")
        elif kind == "error":
            self._console.print(f"[bold red]{message}[/bold red]")
        if self._event_handler is not None:
            self._event_handler(kind, message)

    @staticmethod
    def _format_tool_event(tool_name: str, args: dict[str, Any], result: str) -> str:
        lines = [f"tool: {tool_name}"]
        if tool_name == "run_command":
            command = str(args.get("command", ""))
            cwd = str(args.get("cwd", "."))
            lines.append(f"command: {command}")
            lines.append(f"cwd: {cwd}")
        elif tool_name in {"write_file", "replace_in_file", "read_file", "make_dir", "list_dir"}:
            path = str(args.get("path", ""))
            if path:
                lines.append(f"path: {path}")
        elif tool_name == "web_search":
            lines.append(f"query: {args.get('query', '')}")
        elif tool_name == "open_url":
            lines.append(f"url: {args.get('url', '')}")
        lines.append("result:")
        lines.append(result)
        return "\n".join(lines)

    @staticmethod
    def _format_tool_start(tool_name: str, args: dict[str, Any]) -> str:
        lines = [f"tool: {tool_name}"]
        if tool_name == "run_command":
            lines.append(f"command: {args.get('command', '')}")
            lines.append(f"cwd: {args.get('cwd', '.')}")
        elif tool_name in {"read_file", "write_file", "replace_in_file", "make_dir", "list_dir"}:
            lines.append(f"path: {args.get('path', '.')}")
        elif tool_name == "web_search":
            lines.append(f"query: {args.get('query', '')}")
        elif tool_name == "open_url":
            lines.append(f"url: {args.get('url', '')}")
        return "\n".join(lines)
