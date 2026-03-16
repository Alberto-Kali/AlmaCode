from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from almacode.prompts import build_summary_prompt


def estimate_text_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_message_tokens(message: dict[str, Any]) -> int:
    content = message.get("content", "")
    if isinstance(content, str):
        return estimate_text_tokens(content) + 8
    if isinstance(content, list):
        total = 8
        for item in content:
            if item.get("type") == "text":
                total += estimate_text_tokens(str(item.get("text", "")))
            elif item.get("type") == "image_url":
                total += 600
        return total
    return estimate_text_tokens(str(content)) + 8


@dataclass(slots=True)
class SessionEvent:
    kind: str
    message: str


@dataclass(slots=True)
class PlanStep:
    title: str
    details: str = ""
    status: str = "pending"
    summary: str = ""


@dataclass(slots=True)
class AgentSession:
    summary: str = ""
    recent_history: list[dict[str, Any]] = field(default_factory=list)
    tool_log: list[str] = field(default_factory=list)
    compactions: int = 0
    last_user_task: str = ""
    plan_task: str = ""
    plan_steps: list[PlanStep] = field(default_factory=list)
    active_step_index: int = 0
    research_summary: str = ""

    def append_message(self, message: dict[str, Any]) -> None:
        self.recent_history.append(message)

    def record_tool(self, tool_name: str, result: str) -> None:
        self.tool_log.append(f"{tool_name}: {result[:1200]}")
        self.tool_log = self.tool_log[-20:]

    def token_estimate(self, system_prompt: str, user_message: dict[str, Any]) -> int:
        total = estimate_text_tokens(system_prompt)
        if self.summary:
            total += estimate_text_tokens(self.summary)
        total += estimate_message_tokens(user_message)
        for message in self.recent_history:
            total += estimate_message_tokens(message)
        return total

    def build_messages(
        self,
        *,
        system_prompt: str,
        user_message: dict[str, Any],
        history_tail_messages: int,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        if self.summary.strip():
            messages.append(
                {
                    "role": "system",
                    "content": f"Working memory summary for the current session:\n{self.summary.strip()}",
                }
            )
        tail_history = self.recent_history[-history_tail_messages:] if history_tail_messages > 0 else []
        if tail_history and tail_history[-1] == user_message:
            tail_history = tail_history[:-1]
        if history_tail_messages > 0:
            messages.extend(tail_history)
        messages.append(user_message)
        return messages

    def compact(
        self,
        *,
        backend: Any,
        workspace: str,
        system_prompt: str,
        current_task: str,
        summary_max_tokens: int,
        history_tail_messages: int,
    ) -> str:
        prompt = build_summary_prompt(
            workspace=workspace,
            current_task=current_task,
            summary=self.summary,
            recent_history=self.recent_history,
            tool_log=self.tool_log,
        )
        response = backend.summarize(
            [
                {"role": "system", "content": "Return only a compact working-memory summary."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=summary_max_tokens,
        )
        self.summary = response.strip()
        if history_tail_messages > 0:
            self.recent_history = self.recent_history[-history_tail_messages:]
        else:
            self.recent_history = []
        self.compactions += 1
        return self.summary

    def hard_reset(self, *, current_task: str, history_tail_messages: int) -> None:
        short_log = "; ".join(entry[:120] for entry in self.tool_log[-6:])
        parts = [part for part in [self.summary.strip(), f"Current task: {current_task}", short_log] if part]
        self.summary = "\n".join(parts)[-3000:]
        self.recent_history = self.recent_history[-history_tail_messages:] if history_tail_messages > 0 else []
        self.last_user_task = current_task

    def set_plan(self, *, task: str, steps: list[dict[str, str]], research_summary: str = "") -> None:
        self.plan_task = task
        self.plan_steps = [
            PlanStep(title=step.get("title", "").strip() or f"Step {index}", details=step.get("details", "").strip())
            for index, step in enumerate(steps, start=1)
        ]
        if self.plan_steps:
            self.plan_steps[0].status = "in_progress"
        self.active_step_index = 0
        self.research_summary = research_summary.strip()

    def current_step(self) -> PlanStep | None:
        if not self.plan_steps:
            return None
        if self.active_step_index >= len(self.plan_steps):
            return None
        return self.plan_steps[self.active_step_index]

    def plan_complete(self) -> bool:
        return bool(self.plan_steps) and self.active_step_index >= len(self.plan_steps)

    def complete_current_step(self, summary: str) -> None:
        current = self.current_step()
        if current is None:
            return
        current.status = "completed"
        current.summary = summary.strip()
        self.active_step_index += 1
        next_step = self.current_step()
        if next_step is not None:
            next_step.status = "in_progress"

    def block_current_step(self, summary: str) -> None:
        current = self.current_step()
        if current is None:
            return
        current.status = "blocked"
        current.summary = summary.strip()

    def compact_completed_step(self, *, task: str, history_tail_messages: int) -> None:
        completed = [step for step in self.plan_steps if step.status == "completed" and step.summary]
        if not completed:
            return
        step_lines = [f"- {step.title}: {step.summary}" for step in completed[-6:]]
        parts = [part for part in [self.summary.strip(), f"Current task: {task}", "Completed plan steps:", "\n".join(step_lines)] if part]
        self.summary = "\n".join(parts)[-4000:]
        self.recent_history = self.recent_history[-history_tail_messages:] if history_tail_messages > 0 else []
        self.compactions += 1

    def render_plan(self) -> str:
        if not self.plan_steps:
            return "Plan: [not prepared yet]"
        lines = [f"Task: {self.plan_task or self.last_user_task}"]
        if self.research_summary:
            lines.append(f"Research: {self.research_summary[:220]}")
        for index, step in enumerate(self.plan_steps, start=1):
            marker = {
                "completed": "[x]",
                "in_progress": "[>]",
                "blocked": "[!]",
            }.get(step.status, "[ ]")
            detail = f" - {step.details}" if step.details else ""
            lines.append(f"{marker} {index}. {step.title}{detail}")
        return "\n".join(lines)
