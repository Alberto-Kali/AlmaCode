from pathlib import Path

import pytest
from rich.console import Console

from almacode.agent import CodingAgent, StepLimitReachedError
from almacode.config import AgentConfig
from almacode.llm import ContextOverflowError, LLMResponse
from almacode.session import AgentSession
from almacode.tools import WorkspaceTools


class _OverflowThenAnswerBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.summary_calls = 0

    def complete(self, messages: list[dict[str, object]]) -> LLMResponse:
        self.calls += 1
        if self.calls == 1:
            raise ContextOverflowError(prompt_tokens=4133, context_window=4096, details="too big")
        return LLMResponse(
            content='{"thought":"done","action":{"tool":"final_answer","args":{"answer":"ok"}}}',
            raw={},
        )

    def summarize(self, messages: list[dict[str, object]], max_tokens: int) -> str:
        self.summary_calls += 1
        return "short memory"


def test_agent_recovers_from_context_overflow(tmp_path: Path) -> None:
    backend = _OverflowThenAnswerBackend()
    agent = CodingAgent(
        config=AgentConfig(workspace=tmp_path, server_url="http://127.0.0.1:8080"),
        backend=backend,  # type: ignore[arg-type]
        tools=WorkspaceTools(root=tmp_path, default_timeout=5),
        console=Console(record=True),
    )
    session = AgentSession()
    answer = agent.run("say hi", session=session)
    assert answer == "ok"
    assert backend.summary_calls == 1
    assert session.summary == "short memory"
    assert session.recent_history[-2]["role"] == "user"


class _RepeatingBackend:
    def complete(self, messages: list[dict[str, object]]) -> LLMResponse:
        return LLMResponse(
            content='{"thought":"inspect again","action":{"tool":"list_dir","args":{"path":"."}}}',
            raw={},
        )

    def summarize(self, messages: list[dict[str, object]], max_tokens: int) -> str:
        return "summary"


def test_agent_breaks_repeated_step_loop(tmp_path: Path) -> None:
    agent = CodingAgent(
        config=AgentConfig(workspace=tmp_path, server_url="http://127.0.0.1:8080", max_steps=5),
        backend=_RepeatingBackend(),  # type: ignore[arg-type]
        tools=WorkspaceTools(root=tmp_path, default_timeout=5),
        console=Console(record=True),
    )
    with pytest.raises(StepLimitReachedError):
        agent.run("loop please", session=AgentSession())
