from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AgentConfig:
    model_path: Path
    workspace: Path
    max_steps: int = 18
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 768
    n_ctx: int = 8192
    n_gpu_layers: int = 0
    n_threads: int | None = None
    command_timeout: int = 60
    chat_format: str | None = None
    verbose: bool = False
    system_note: str = ""

