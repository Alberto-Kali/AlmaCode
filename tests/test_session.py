from pathlib import Path

from almacode.session import AgentSession


class _SummaryBackend:
    def summarize(self, messages: list[dict[str, object]], max_tokens: int) -> str:
        return f"summary up to {max_tokens}"


def test_session_compact_keeps_summary_and_tail() -> None:
    session = AgentSession()
    for index in range(8):
        session.append_message({"role": "user", "content": f"message {index}"})
    session.record_tool("list_dir", "ok")
    summary = session.compact(
        backend=_SummaryBackend(),
        workspace=str(Path.cwd()),
        system_prompt="system",
        current_task="task",
        summary_max_tokens=200,
        history_tail_messages=2,
    )
    assert "summary" in summary
    assert len(session.recent_history) == 2
    assert session.compactions == 1


def test_session_hard_reset_preserves_short_memory() -> None:
    session = AgentSession(summary="old summary")
    for index in range(5):
        session.record_tool("run_command", f"stdout {index}")
        session.append_message({"role": "assistant", "content": f"reply {index}"})
    session.hard_reset(current_task="finish task", history_tail_messages=2)
    assert "finish task" in session.summary
    assert len(session.recent_history) == 2
