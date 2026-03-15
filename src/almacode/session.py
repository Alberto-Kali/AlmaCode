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
class AgentSession:
    summary: str = ""
    recent_history: list[dict[str, Any]] = field(default_factory=list)
    tool_log: list[str] = field(default_factory=list)
    compactions: int = 0

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
        if history_tail_messages > 0:
            messages.extend(self.recent_history[-history_tail_messages:])
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
